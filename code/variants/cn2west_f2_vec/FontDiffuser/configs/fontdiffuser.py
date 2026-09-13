import os
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Training config for FontDiffuser.")
    ################# Experience #################
    parser.add_argument("--seed", type=int, default=3407, help="A seed for reproducible training.")
    parser.add_argument("--experience_name", type=str, default="fontdiffuer_training")
    parser.add_argument("--data_root", type=str, default=None,
                        help="The font dataset root path.",)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--logging_dir", type=str, default="logs",
                        help=("[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
                              " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."))

    # Model
    parser.add_argument("--resolution", type=int, default=96,
                        help="The resolution for input images, all the images in the train/validation \
                            dataset will be resized to this.")
    parser.add_argument("--unet_channels", type=tuple, default=(64, 128, 256, 512),
                        help="The channels of the UNet.")
    parser.add_argument("--style_image_size", type=int, default=96, help="The size of style images.")
    parser.add_argument("--content_image_size", type=int, default=96, help="The size of content images.")
    parser.add_argument("--content_encoder_downsample_size", type=int, default=3,
                        help="The downsample size of the content encoder.")
    parser.add_argument("--channel_attn", type=bool, default=True, help="Whether to use the se attention.",)
    parser.add_argument("--content_start_channel", type=int, default=64,
                        help="The channels of the fisrt layer output of content encoder.",)
    parser.add_argument("--style_start_channel", type=int, default=64,
                        help="The channels of the fisrt layer output of content encoder.",)
    parser.add_argument("--rsi_source", choices=("delta", "official"), default="delta")
    parser.add_argument("--delta_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--delta_tau", type=float, default=0.07)
    parser.add_argument("--delta_eps_alpha", type=float, default=0.01)
    parser.add_argument("--delta_k_max", type=int, default=10)
    parser.add_argument("--delta_k_top", type=int, default=10)
    parser.add_argument("--delta_mode", choices=("soft", "topk", "threshold"), default="topk")
    parser.add_argument("--ec_cache_path", type=str, default="artifacts/f0/ec_multiscale_f0")
    parser.add_argument("--encoder_runtime", choices=("cache_only", "online"), default="cache_only")
    # F1/F2/F3 matched arms. source_drop applies to WHATEVER structure source is
    # active (official Ec or delta) -- the old delta-only drop silently un-matched
    # the official arm against the delta arm.
    parser.add_argument("--arm", choices=("F1", "F2", "F3", "F3b", "F2P", "F3bP", "F2VEC"), required=True)
    parser.add_argument("--vec_warmup_steps", type=int, default=5000)
    parser.add_argument("--vec_unet_scale", type=float, default=0.15)
    parser.add_argument("--vec_consist", type=float, default=0.1)
    parser.add_argument("--source_drop", type=float, default=0.25)
    parser.add_argument("--support", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--support_drop", type=float, default=0.20)
    parser.add_argument("--support_k", type=int, default=8)
    parser.add_argument("--support_bank", type=str, default=None,
                        help="JSON manifest {char_cp: [support_cp, ...]}; required when --support.")
    parser.add_argument("--parity_check", action=argparse.BooleanOptionalAction, default=True,
                        help="Assert step-0 RSI output is bit-identical to the F0 raw-skip path.")
    parser.add_argument("--nshot_min", type=int, default=1)
    parser.add_argument("--nshot_max", type=int, default=8)
    parser.add_argument("--eval_refs", nargs="+", default=list("永和书风骨韵天地"))
    parser.add_argument("--es_cache_path", type=str, default="artifacts/f0/es_spatial_f0")
    parser.add_argument("--split_manifest", type=str, default="manifests/split_v3_228_16_16.json")
    parser.add_argument("--excluded", nargs="*", default=["FZXianZTJW"])
    parser.add_argument("--freeze_encoders", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--config_path", type=str, default=None)

    # Training
    parser.add_argument("--phase_2", action="store_true", help="Training in phase 2 using SCR module.")
    parser.add_argument("--phase_1_ckpt_dir", type=str, default=None, help="The trained ckpt directory during phase 1.")
    ## SCR
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--mode", type=str, default="refinement")
    parser.add_argument("--scr_image_size", type=int, default=96)
    parser.add_argument("--scr_ckpt_path", type=str, default=None)
    parser.add_argument("--num_neg", type=int, default=16, help="Number of negative samples.")
    parser.add_argument("--nce_layers", type=str, default='0,1,2,3')
    parser.add_argument("--sc_coefficient", type=float, default=0.01)
    ## train batch size
    parser.add_argument("--train_batch_size", type=int, default=8,
                        help="Batch size (per device) for the training dataloader.")
    ## loss coefficient
    parser.add_argument("--perceptual_coefficient", type=float, default=0.01)
    parser.add_argument("--offset_coefficient", type=float, default=0.5)
    ## step
    parser.add_argument("--max_train_steps", type=int, default=80000,
                        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",)
    parser.add_argument("--ckpt_interval", type=int,default=5000, help="The checkpoint interval.")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from a checkpoint dir (global_step_* or last_state) with trainer_state.pt.")
    parser.add_argument("--state_interval", type=int, default=1000,
                        help="Interval to save crash-recovery last_state (weights+optimizer+RNG).")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--log_interval", type=int, default=100, help="The log interval of training.")
    ## learning rate
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--scale_lr", action="store_true", default=False,
                        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")
    parser.add_argument("--lr_scheduler", type=str, default="linear",
                        help="The scheduler type to use. Choose between 'linear', 'cosine', \
                            'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'")
    parser.add_argument("--lr_warmup_steps", type=int, default=2000,
                        help="Number of steps for the warmup in the lr scheduler.")
    ## classifier-free
    parser.add_argument("--drop_prob", type=float, default=0.1, help="The uncondition training drop out probability.")
    ## scheduler
    parser.add_argument("--beta_scheduler", type=str, default="scaled_linear", help="The beta scheduler for DDPM.")
    ## optimizer
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"],
                        help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires \
                            PyTorch >= 1.10. and an Nvidia Ampere GPU.")

    # Sampling
    parser.add_argument("--algorithm_type", type=str, default="dpmsolver++", help="Algorithm for sampleing.")
    parser.add_argument("--guidance_type", type=str, default="classifier-free", help="Guidance type of sampling.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale of the classifier-free mode.")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Sampling step.")
    parser.add_argument("--model_type", type=str, default="noise", help="model_type for sampling.")
    parser.add_argument("--order", type=int, default=2, help="The order of the dpmsolver.")
    parser.add_argument("--skip_type", type=str, default="time_uniform", help="Skip type of dpmsolver.")
    parser.add_argument("--method", type=str, default="multistep", help="Multistep of dpmsolver.")
    parser.add_argument("--correcting_x0_fn", type=str, default=None, help="correcting_x0_fn of dpmsolver.")
    parser.add_argument("--t_start", type=str, default=None, help="t_start of dpmsolver.")
    parser.add_argument("--t_end", type=str, default=None, help="t_end of dpmsolver.")

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    return parser
