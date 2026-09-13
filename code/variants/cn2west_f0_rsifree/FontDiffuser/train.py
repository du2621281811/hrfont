import os
import math
import time
import logging
import glob
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler

from dataset.font_dataset import FontDataset
from dataset.collate_fn import CollateFN
from configs.fontdiffuser import get_parser
from src import (FontDiffuserModel,
                 ContentPerceptualLoss,
                 build_unet,
                 build_style_encoder,
                 build_content_encoder,
                 build_ddpm_scheduler,
                 build_scr)
from utils import (save_args_to_yaml,
                   x0_from_epsilon, 
                   reNormalize_img, 
                   normalize_mean_std)


logger = get_logger(__name__)

def get_args():
    parser = get_parser()
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)

    return args


def _find_latest_resume(output_dir: str) -> str | None:
    """Prefer last_state/, else highest global_step_* with trainer_state.pt."""
    last = Path(output_dir) / "last_state" / "trainer_state.pt"
    if last.is_file():
        return str(last.parent)
    cands = []
    for p in Path(output_dir).glob("global_step_*"):
        if (p / "trainer_state.pt").is_file() and (p / "unet.pth").is_file():
            try:
                step = int(p.name.split("_")[-1])
            except ValueError:
                continue
            cands.append((step, str(p)))
    if not cands:
        return None
    cands.sort()
    return cands[-1][1]


def _save_trainable_ckpt(model, save_dir: str, optimizer, lr_scheduler, global_step, scaler=None):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.unet.state_dict(), f"{save_dir}/unet.pth")
    torch.save(model.style_encoder.state_dict(), f"{save_dir}/style_encoder.pth")
    torch.save(model.content_encoder.state_dict(), f"{save_dir}/content_encoder.pth")
    state = {
        "global_step": global_step,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_seed_state": None,
    }
    try:
        import numpy as np
        state["numpy_seed_state"] = np.random.get_state()
    except Exception:
        pass
    try:
        import random as py_random
        state["python_rng"] = py_random.getstate()
    except Exception:
        pass
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    torch.save(state, f"{save_dir}/trainer_state.pt")


def _load_trainable_ckpt(model, resume_dir: str, optimizer, lr_scheduler, scaler=None):
    model.unet.load_state_dict(torch.load(f"{resume_dir}/unet.pth", map_location="cpu"))
    model.style_encoder.load_state_dict(torch.load(f"{resume_dir}/style_encoder.pth", map_location="cpu"))
    model.content_encoder.load_state_dict(torch.load(f"{resume_dir}/content_encoder.pth", map_location="cpu"))
    state_path = f"{resume_dir}/trainer_state.pt"
    global_step = 0
    if os.path.isfile(state_path):
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        global_step = int(state.get("global_step", 0))
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if "lr_scheduler" in state:
            lr_scheduler.load_state_dict(state["lr_scheduler"])
        if scaler is not None and "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        if state.get("torch_rng") is not None:
            torch.set_rng_state(state["torch_rng"])
        if state.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if state.get("numpy_seed_state") is not None:
            import numpy as np
            np.random.set_state(state["numpy_seed_state"])
        if state.get("python_rng") is not None:
            import random as py_random
            py_random.setstate(state["python_rng"])
    return global_step


def main():

    args = get_args()

    logging_dir = f"{args.output_dir}/{args.logging_dir}"

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    logging.basicConfig(
        filename=f"{args.output_dir}/fontdiffuser_training.log",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    # Ser training seed
    if args.seed is not None:
        set_seed(args.seed)

    # Load model and noise_scheduler
    unet = build_unet(args=args)
    style_encoder = build_style_encoder(args=args)
    content_encoder = build_content_encoder(args=args)
    noise_scheduler = build_ddpm_scheduler(args)
    # cn2west_f0_rsifree: load official/P1 with strict=False; drop RSI/DCN-only keys.
    if getattr(args, "phase_1_ckpt_dir", None):
        import json as _json
        from pathlib import Path as _Path
        unet_sd = torch.load(f"{args.phase_1_ckpt_dir}/unet.pth", map_location="cpu")
        miss, unexp = unet.load_state_dict(unet_sd, strict=False)
        # Allowlist: P1 StyleRSI keys that F0 intentionally omits; unexpected should be empty.
        rsi_prefixes = ("sc_interpreter_offsets", "dcn_deforms")
        dropped = [k for k in unexp if any(s in k for s in rsi_prefixes)]
        other_unexp = [k for k in unexp if k not in dropped]
        # Also record P1 keys present in ckpt but never requested (strict=False does not list them).
        model_keys = set(unet.state_dict().keys())
        ckpt_only = sorted(k for k in unet_sd.keys() if k not in model_keys)
        allow = {
            "phase_1_ckpt_dir": args.phase_1_ckpt_dir,
            "missing_keys": list(miss),
            "unexpected_keys": list(unexp),
            "ckpt_only_keys": ckpt_only,
            "rsi_dcn_dropped_prefixes": list(rsi_prefixes),
            "other_unexpected_keys": other_unexp,
            "note": "F0 StyleUpBlockNoRSI: load overlapping resnet/attention/upsample; drop RSI/DCN.",
        }
        out_allow = _Path(args.output_dir) / "p1_load_allowlist.json"
        out_allow.parent.mkdir(parents=True, exist_ok=True)
        out_allow.write_text(_json.dumps(allow, indent=2) + "\n", encoding="utf-8")
        if other_unexp:
            raise RuntimeError(f"unexpected non-RSI keys when loading P1 into F0: {other_unexp[:20]}")
        # Missing keys should be empty: F0 submodules that exist must be in P1 under same names.
        if miss:
            logger.warning(f"F0 missing keys after P1 load ({len(miss)}): {miss[:10]}...")
        style_encoder.load_state_dict(torch.load(f"{args.phase_1_ckpt_dir}/style_encoder.pth", map_location="cpu"))
        content_encoder.load_state_dict(torch.load(f"{args.phase_1_ckpt_dir}/content_encoder.pth", map_location="cpu"))
        logger.info(
            f"Loaded P1 weights from {args.phase_1_ckpt_dir} (strict=False); "
            f"ckpt_only={len(ckpt_only)} missing={len(miss)} allowlist={out_allow}"
        )

    model = FontDiffuserModel(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder)

    # Build content perceptaual Loss
    perceptual_loss = ContentPerceptualLoss()

    # Load SCR module for supervision
    if args.phase_2:
        scr = build_scr(args=args)
        scr.load_state_dict(torch.load(args.scr_ckpt_path))
        scr.requires_grad_(False)

    # A-protocol: native 96 PNG — ToTensor + Normalize only (no Resize).
    def _assert_native_size(img):
        w, h = img.size
        exp = args.resolution if isinstance(args.resolution, int) else args.resolution[0]
        if (w, h) != (exp, exp):
            raise ValueError(f"expected native {exp}x{exp}, got {img.size}")
        return img

    content_transforms = transforms.Compose(
        [_assert_native_size,
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    style_transforms = transforms.Compose(
        [_assert_native_size,
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    target_transforms = transforms.Compose(
        [_assert_native_size,
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    train_font_dataset = FontDataset(
        args=args,
        phase='train', 
        transforms=[
            content_transforms, 
            style_transforms, 
            target_transforms],
        scr=args.phase_2)
    sampler = None
    if getattr(train_font_dataset, "sample_weights", None):
        gen = torch.Generator()
        # Rank-offset so DDP ranks do not draw the identical weighted stream.
        gen.manual_seed(int(args.seed) + int(accelerator.process_index))
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.as_tensor(train_font_dataset.sample_weights, dtype=torch.double),
            num_samples=len(train_font_dataset.sample_weights),
            replacement=True,
            generator=gen,
        )
        logger.info(
            f"v0913_clean WeightedRandomSampler on {len(train_font_dataset)} pairs "
            f"from {args.v0913_clean_map}"
        )
    train_dataloader = torch.utils.data.DataLoader(
        train_font_dataset,
        shuffle=sampler is None,
        sampler=sampler,
        batch_size=args.train_batch_size,
        collate_fn=CollateFN(),
    )
    
    # Build optimizer and learning rate
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,)

    # Accelerate preparation
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler)
    ## move scr module to the target deivces
    if args.phase_2:
        scr = scr.to(accelerator.device)

    # Resolve resume (crash recovery / watchdog) — same hyperparams, no degradation.
    resume_dir = getattr(args, "resume_from", None) or None
    if resume_dir in (None, "", "auto"):
        resume_dir = _find_latest_resume(args.output_dir)
    global_step = 0
    if resume_dir:
        # unwrap for attribute access after prepare
        raw = accelerator.unwrap_model(model)
        global_step = _load_trainable_ckpt(raw, resume_dir, optimizer, lr_scheduler)
        logger.info(f"Resumed from {resume_dir} at global_step={global_step}")
        print(f"Resumed from {resume_dir} at global_step={global_step}", flush=True)

    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(args.experience_name)
        save_args_to_yaml(args=args, output_file=f"{args.output_dir}/{args.experience_name}_config.yaml")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process, initial=global_step)
    progress_bar.set_description("Steps")

    # Convert to the training epoch
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    state_interval = int(getattr(args, "state_interval", 1000) or 1000)

    for epoch in range(num_train_epochs):
        train_loss = 0.0
        for step, samples in enumerate(train_dataloader):
            if global_step >= args.max_train_steps:
                break
            model.train()
            content_images = samples["content_image"]
            style_images = samples["style_image"]
            target_images = samples["target_image"]
            nonorm_target_images = samples["nonorm_target_image"]
            
            with accelerator.accumulate(model):
                # Sample noise that we'll add to the samples
                noise = torch.randn_like(target_images)
                bsz = target_images.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=target_images.device)
                timesteps = timesteps.long()

                # Add noise to the target_images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_target_images = noise_scheduler.add_noise(target_images, noise, timesteps)

                # Classifier-free training strategy
                context_mask = torch.bernoulli(torch.zeros(bsz) + args.drop_prob)
                for i, mask_value in enumerate(context_mask):
                    if mask_value==1:
                        content_images[i, :, :, :] = 1
                        style_images[i, :, :, :] = 1

                # Predict the noise residual and compute loss
                noise_pred, offset_out_sum = model(
                    x_t=noisy_target_images, 
                    timesteps=timesteps, 
                    style_images=style_images,
                    content_images=content_images,
                    content_encoder_downsample_size=args.content_encoder_downsample_size)
                diff_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                offset_loss = offset_out_sum / 2
                
                # output processing for content perceptual loss
                pred_original_sample_norm = x0_from_epsilon(
                    scheduler=noise_scheduler,
                    noise_pred=noise_pred,
                    x_t=noisy_target_images,
                    timesteps=timesteps)
                pred_original_sample = reNormalize_img(pred_original_sample_norm)
                norm_pred_ori = normalize_mean_std(pred_original_sample)
                norm_target_ori = normalize_mean_std(nonorm_target_images)
                percep_loss = perceptual_loss.calculate_loss(
                    generated_images=norm_pred_ori,
                    target_images=norm_target_ori,
                    device=target_images.device)
                
                loss = diff_loss + \
                        args.perceptual_coefficient * percep_loss + \
                            args.offset_coefficient * offset_loss
                
                if args.phase_2:
                    neg_images = samples["neg_images"]
                    # sc loss
                    sample_style_embeddings, pos_style_embeddings, neg_style_embeddings = scr(
                        pred_original_sample_norm, 
                        target_images, 
                        neg_images, 
                        nce_layers=args.nce_layers)
                    sc_loss = scr.calculate_nce_loss(
                        sample_s=sample_style_embeddings,
                        pos_s=pos_style_embeddings,
                        neg_s=neg_style_embeddings)
                    loss += args.sc_coefficient * sc_loss

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    raw = accelerator.unwrap_model(model)
                    if global_step % args.ckpt_interval == 0:
                        save_dir = f"{args.output_dir}/global_step_{global_step}"
                        _save_trainable_ckpt(raw, save_dir, optimizer, lr_scheduler, global_step)
                        logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}] Save the checkpoint on global step {global_step}")
                        print("Save the checkpoint on global step {}".format(global_step), flush=True)
                    if state_interval > 0 and global_step % state_interval == 0:
                        _save_trainable_ckpt(raw, f"{args.output_dir}/last_state", optimizer, lr_scheduler, global_step)
                        print(f"Save last_state at global step {global_step}", flush=True)

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if global_step % args.log_interval == 0:
                logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}] Global Step {global_step} => train_loss = {loss}")
            progress_bar.set_postfix(**logs)
            
            # Quit
            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    # Final crash-recovery snapshot at completion
    if accelerator.is_main_process:
        raw = accelerator.unwrap_model(model)
        _save_trainable_ckpt(raw, f"{args.output_dir}/last_state", optimizer, lr_scheduler, global_step)
        done = Path(args.output_dir) / "DONE.json"
        done.write_text(
            f'{{"global_step": {global_step}, "max_train_steps": {args.max_train_steps}, "status": "completed"}}\n',
            encoding="utf-8",
        )
        print(f"TRAINING COMPLETED at step {global_step}", flush=True)

    accelerator.end_training()

if __name__ == "__main__":
    main()
