import json
import random
from pathlib import Path
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
        self.args = args
        self.root = Path(args.data_root)
        self.phase = phase
        self.scr = scr
        self.nshot_min = args.nshot_min
        self.nshot_max = args.nshot_max
        self.eval_refs = list(args.eval_refs)
        self.split_manifest = Path(args.split_manifest)
        self.excluded = set(args.excluded)
        if self.scr:
            self.num_neg = args.num_neg

        self.sample_weights = None
        self.get_path()
        self.transforms = transforms
        self.nonorm_transforms = get_nonorm_transform(args.resolution)

    def get_path(self):
        self.target_images = []
        self.target_by_font_char = {}
        self.style_by_font_char = {}
        target_image_dir = self.root / self.phase / "TargetImage"
        style_image_dir = self.root / self.phase / "StyleImage"
        if not style_image_dir.is_dir():
            raise FileNotFoundError(f"StyleImage required: {style_image_dir}")

        manifest = json.loads(self.split_manifest.read_text(encoding="utf-8"))
        expected = set(manifest["stems"][self.phase])
        physical = {p.name for p in target_image_dir.iterdir() if p.is_dir()}
        physical_style = {p.name for p in style_image_dir.iterdir() if p.is_dir()}
        if physical != expected or physical_style != expected or physical & self.excluded:
            raise RuntimeError(
                f"split manifest mismatch for {self.phase}: "
                f"missing={sorted(expected - physical)} extra={sorted(physical - expected)} "
                f"style_missing={sorted(expected - physical_style)} "
                f"style_extra={sorted(physical_style - expected)} "
                f"excluded={sorted(physical & self.excluded)}"
            )

        for font in sorted(physical):
            target_map = self._char_map(target_image_dir / font, font)
            style_map = self._char_map(style_image_dir / font, font)
            if not style_map:
                raise FileNotFoundError(f"StyleImage missing/empty for font={font}")
            self.target_by_font_char[font] = target_map
            self.style_by_font_char[font] = style_map
            self.target_images.extend(str(target_map[cp]) for cp in sorted(target_map))

        map_dir = getattr(self.args, "v0913_clean_map", None) or ""
        if map_dir:
            self._filter_clean_pairs(map_dir)

        if not self.target_images:
            raise RuntimeError(f"no TargetImage PNGs under {target_image_dir}")

    @staticmethod
    def _char_map(directory: Path, font: str) -> dict[str, Path]:
        if not directory.is_dir():
            return {}
        result = {}
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() != ".png":
                continue
            prefix = f"{font}+"
            if not path.stem.startswith(prefix):
                continue
            result[path.stem[len(prefix):]] = path
        return result

    def _filter_clean_pairs(self, map_dir: str):
        import csv
        import json
        tsv = Path(map_dir) / f"pairs_{self.phase}.tsv"
        rows = list(csv.DictReader(tsv.open(encoding="utf-8", newline=""), delimiter="\t"))
        allowed = {(row["font"], row["cp"]) for row in rows}
        filtered = []
        for path in self.target_images:
            p = Path(path)
            font = p.parent.name
            cp = p.stem[len(font) + 1:]
            if (font, cp) in allowed:
                filtered.append(path)
        if len(filtered) != len(allowed):
            raise RuntimeError(
                f"v0913_clean {self.phase} pairs={len(allowed)} files={len(filtered)}"
            )
        self.target_images = filtered
        if self.phase == "train":
            table = json.loads((Path(map_dir) / "sample_weights.json").read_text(encoding="utf-8"))["pair_weight"]
            group = {(row["font"], row["cp"]): row["script_group"] for row in rows}
            weights = []
            for path in self.target_images:
                p = Path(path)
                font = p.parent.name
                cp = p.stem[len(font) + 1:]
                weights.append(float(table[group[(font, cp)]]))
            self.sample_weights = weights

    def _open_content(self, content_cp: str) -> Image.Image:
        path = self.root / self.phase / "ContentImage" / f"{content_cp}.png"
        if not path.is_file():
            raise FileNotFoundError(f"ContentImage missing: {path}")
        return Image.open(path).convert("RGB")

    def content_path(self, content_cp: str) -> str:
        path = self.root / self.phase / "ContentImage" / f"{content_cp}.png"
        if not path.exists():
            raise FileNotFoundError(f"ContentImage missing: {path}")
        return str(path)

    def target_path(self, font: str, content_cp: str) -> str:
        try:
            return str(self.target_by_font_char[font][content_cp])
        except KeyError as exc:
            raise FileNotFoundError(f"TargetImage missing for {font}+{content_cp}") from exc

    def __getitem__(self, index):
        target_image_path = self.target_images[index]
        target_path = Path(target_image_path)
        font = target_path.parent.name
        content = target_path.stem[len(font) + 1:]

        content_image = self._open_content(content)

        # One episode R for Es, α query, and α prototypes. Do not draw a second set.
        style_map = self.style_by_font_char[font]
        if self.phase == "train":
            n = random.randint(self.nshot_min, self.nshot_max)
            ref_chars = random.sample(sorted(style_map), n)
        else:
            ref_chars = [f"u{ord(ch):04X}" for ch in self.eval_refs]
            missing = [cp for cp in ref_chars if cp not in style_map]
            if missing:
                raise FileNotFoundError(f"fixed refs missing for {font}: {missing}")
        ref_image_paths = [str(style_map[cp]) for cp in ref_chars]

        target_image = Image.open(target_image_path).convert("RGB")
        nonorm_target_image = self.nonorm_transforms(target_image)

        if self.transforms is not None:
            content_image = self.transforms[0](content_image)
            target_image = self.transforms[2](target_image)

        sample = {
            "content_image": content_image,
            "target_image": target_image,
            "target_image_path": target_image_path,
            "nonorm_target_image": nonorm_target_image,
            "font_stem": font,
            "char_cp": content,
            "split": self.phase,
            "ref_chars": ref_chars,
            "ref_image_paths": ref_image_paths,
        }

        if self.scr:
            # SCR not used in E1 FT-v2; kept for API compatibility with PNG paths.
            style_list = [s for s in self.style_by_font_char if s != font]
            choose_neg_names = []
            for _ in range(self.num_neg):
                choose_style = random.choice(style_list)
                choose_neg_names.append(
                    str(self.root / "train" / "TargetImage" / choose_style / f"{choose_style}+{content}.png")
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
