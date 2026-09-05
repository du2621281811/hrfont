import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def get_nonorm_transform(resolution):
    """No resize — A-protocol PNGs are already resolution×resolution."""
    def _check_and_tensor(img: Image.Image):
        if img.size != (resolution, resolution):
            raise ValueError(
                f"expected native {resolution}x{resolution}, got {img.size}"
            )
        return transforms.functional.to_tensor(img)

    return _check_and_tensor


class FontDataset(Dataset):
    """FontDiffuser dataset patched for cn2west A-protocol.

    - Target/Style/Content are RGB PNG (no JPG).
    - ContentImage/{cp}.png where cp is the codepoint token from the target name.
    - Style is sampled from StyleImage/<font>/ (CN style pool), not from TargetImage.
    """

    def __init__(self, args, phase, transforms=None, scr=False):
        super().__init__()
        self.root = args.data_root
        self.phase = phase
        self.scr = scr
        if self.scr:
            self.num_neg = args.num_neg

        self.get_path()
        self.transforms = transforms
        self.nonorm_transforms = get_nonorm_transform(args.resolution)

    def get_path(self):
        self.target_images = []
        self.style_to_images = {}
        target_image_dir = f"{self.root}/{self.phase}/TargetImage"
        style_image_dir = f"{self.root}/{self.phase}/StyleImage"
        if not os.path.isdir(style_image_dir):
            raise FileNotFoundError(f"StyleImage required: {style_image_dir}")

        for style in sorted(os.listdir(target_image_dir)):
            style_target_dir = f"{target_image_dir}/{style}"
            if not os.path.isdir(style_target_dir):
                continue
            for img in sorted(os.listdir(style_target_dir)):
                if not img.lower().endswith(".png"):
                    continue
                self.target_images.append(f"{style_target_dir}/{img}")

            style_src = f"{style_image_dir}/{style}"
            images_related_style = []
            if os.path.isdir(style_src):
                for img in sorted(os.listdir(style_src)):
                    if img.lower().endswith(".png"):
                        images_related_style.append(f"{style_src}/{img}")
            if not images_related_style:
                raise FileNotFoundError(
                    f"CN StyleImage missing/empty for font={style}: {style_src}"
                )
            self.style_to_images[style] = images_related_style

        if not self.target_images:
            raise RuntimeError(f"no TargetImage PNGs under {target_image_dir}")

    @staticmethod
    def _open_content(root, phase, content_cp: str) -> Image.Image:
        base = f"{root}/{phase}/ContentImage/{content_cp}"
        for ext in (".png", ".jpg", ".jpeg"):
            path = base + ext
            if os.path.isfile(path):
                return Image.open(path).convert("RGB")
        raise FileNotFoundError(f"ContentImage missing for {content_cp} under {root}/{phase}/ContentImage")

    def __getitem__(self, index):
        target_image_path = self.target_images[index]
        target_image_name = target_image_path.split("/")[-1]
        style, content = target_image_name.split(".")[0].split("+")

        content_image = self._open_content(self.root, self.phase, content)

        images_related_style = self.style_to_images[style]
        style_image_path = random.choice(images_related_style)
        style_image = Image.open(style_image_path).convert("RGB")

        target_image = Image.open(target_image_path).convert("RGB")
        nonorm_target_image = self.nonorm_transforms(target_image)

        if self.transforms is not None:
            content_image = self.transforms[0](content_image)
            style_image = self.transforms[1](style_image)
            target_image = self.transforms[2](target_image)

        sample = {
            "content_image": content_image,
            "style_image": style_image,
            "target_image": target_image,
            "target_image_path": target_image_path,
            "nonorm_target_image": nonorm_target_image,
        }

        if self.scr:
            # SCR not used in E1 FT-v2; kept for API compatibility with PNG paths.
            style_list = [s for s in self.style_to_images.keys() if s != style]
            choose_neg_names = []
            for _ in range(self.num_neg):
                choose_style = random.choice(style_list)
                choose_neg_names.append(
                    f"{self.root}/train/TargetImage/{choose_style}/{choose_style}+{content}.png"
                )
            neg_images = None
            for i, neg_name in enumerate(choose_neg_names):
                neg_image = Image.open(neg_name).convert("RGB")
                if self.transforms is not None:
                    neg_image = self.transforms[2](neg_image)
                if i == 0:
                    neg_images = neg_image[None, :, :, :]
                else:
                    neg_images = torch.cat([neg_images, neg_image[None, :, :, :]], dim=0)
            sample["neg_images"] = neg_images

        return sample

    def __len__(self):
        return len(self.target_images)
