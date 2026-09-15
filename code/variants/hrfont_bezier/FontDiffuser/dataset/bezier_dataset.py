"""Vector sidecar dataset for the HR-Font Bezier MVP."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


REQUIRED_KEYS = frozenset({
    "points", "point_types", "contour_ids", "segments", "segment_contours", "metrics"
})


def load_outline(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Bezier sidecar missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        missing = REQUIRED_KEYS - set(payload.files)
        if missing:
            raise ValueError(f"{path}: missing vector fields {sorted(missing)}")
        points = np.asarray(payload["points"], dtype=np.float32)
        point_types = np.asarray(payload["point_types"], dtype=np.float32)
        contour_ids = np.asarray(payload["contour_ids"], dtype=np.int64)
        segments = np.asarray(payload["segments"], dtype=np.int64)
        segment_contours = np.asarray(payload["segment_contours"], dtype=np.int64)
        metrics = np.asarray(payload["metrics"], dtype=np.float32)
        adjacency = np.asarray(payload["adjacent_segments"], dtype=np.int64) if "adjacent_segments" in payload else np.zeros((0, 2), dtype=np.int64)

    if points.ndim != 2 or points.shape[1] != 2 or not len(points):
        raise ValueError(f"{path}: points must be non-empty [N,2]")
    if segments.ndim != 2 or segments.shape[1] != 4 or not len(segments):
        raise ValueError(f"{path}: segments must be non-empty cubic [S,4]")
    if segments.min() < 0 or segments.max() >= len(points):
        raise ValueError(f"{path}: segment index outside point array")
    if segment_contours.shape != (len(segments),):
        raise ValueError(f"{path}: segment_contours length mismatch")
    if point_types.shape != (len(points),) or contour_ids.shape != (len(points),):
        raise ValueError(f"{path}: point metadata length mismatch")
    if metrics.shape != (3,):
        raise ValueError(f"{path}: metrics must be [advance, lsb, rsb] normalized by UPM")
    if not np.isfinite(points).all() or not np.isfinite(metrics).all():
        raise ValueError(f"{path}: non-finite outline data")

    contour_scale = max(1, int(contour_ids.max(initial=0)))
    point_features = np.concatenate([
        points,
        (point_types[:, None] == 0).astype(np.float32),  # on-curve endpoint
        (point_types[:, None] == 1).astype(np.float32),  # off-curve control
        (contour_ids[:, None] / contour_scale).astype(np.float32),
        np.ones((len(points), 1), dtype=np.float32),
    ], axis=1)
    return {
        "point_features": torch.from_numpy(point_features),
        "segments": torch.from_numpy(segments),
        "segment_contours": torch.from_numpy(segment_contours),
        "metrics": torch.from_numpy(metrics),
        "adjacent_segments": torch.from_numpy(adjacency),
        "path": str(path),
    }


class BezierFontDataset(Dataset):
    """Add neutral vector templates and target metrics to a raster FontDataset.

    Sidecar layout mirrors the PNG roles::

      vector_root/<split>/ContentOutline/<cp>.npz
      vector_root/<split>/TargetOutline/<font>/<font>+<cp>.npz

    Target outlines are retained for future Chamfer/topology objectives.  The
    fixed-topology MVP only needs their metrics because image/SDF supervision
    does not assume neutral-to-target point correspondence.
    """

    def __init__(self, raster_dataset, vector_root: str | Path):
        self.raster_dataset = raster_dataset
        self.vector_root = Path(vector_root)

    def __len__(self):
        return len(self.raster_dataset)

    def __getitem__(self, index):
        sample = self.raster_dataset[index]
        split, font, cp = sample["split"], sample["font_stem"], sample["char_cp"]
        neutral_path = self.vector_root / split / "ContentOutline" / f"{cp}.npz"
        target_path = self.vector_root / split / "TargetOutline" / font / f"{font}+{cp}.npz"
        neutral = load_outline(neutral_path)
        target = load_outline(target_path)
        # Equal counts are only a diagnostic, not point correspondence or proof
        # of equal topology.
        sample["topology_count_match"] = neutral["segments"].shape == target["segments"].shape
        sample["bezier"] = neutral
        sample["target_bezier"] = target
        sample["target_bezier_path"] = target["path"]
        sample["target_metrics"] = target["metrics"]
        return sample


def _pad_2d(items, value=0.0):
    width = max(x.shape[0] for x in items)
    result = items[0].new_full((len(items), width, items[0].shape[1]), value)
    mask = torch.zeros(len(items), width, dtype=torch.bool)
    for i, item in enumerate(items):
        result[i, :item.shape[0]] = item
        mask[i, :item.shape[0]] = True
    return result, mask


class BezierCollateFN:
    def __call__(self, batch):
        result = {}
        vector = [sample.pop("bezier") for sample in batch]
        target_vector = [sample.pop("target_bezier") for sample in batch]
        for key in batch[0]:
            values = [sample[key] for sample in batch]
            result[key] = torch.stack(values) if isinstance(values[0], torch.Tensor) else values

        result["point_features"], result["point_mask"] = _pad_2d(
            [item["point_features"] for item in vector]
        )
        result["segment_indices"], result["segment_mask"] = _pad_2d(
            [item["segments"] for item in vector], value=-1
        )
        result["adjacent_segments"], result["adjacency_mask"] = _pad_2d(
            [item["adjacent_segments"] for item in vector], value=-1
        )
        result["target_point_features"], result["target_point_mask"] = _pad_2d(
            [item["point_features"] for item in target_vector]
        )
        result["target_segment_indices"], result["target_segment_mask"] = _pad_2d(
            [item["segments"] for item in target_vector], value=-1
        )
        # FontDataset is black ink on white in [0,1]; the vector rasterizer emits
        # ink occupancy (1=ink), so invert the grayscale target explicitly.
        result["target_raster_01"] = 1.0 - result["nonorm_target_image"][:, :1]
        return result
