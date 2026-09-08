#!/usr/bin/env python3
"""Export only stroke feature and style adapter weights from a full checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def extract_prefixed(state_dict, marker):
    extracted = {}
    for key, value in state_dict.items():
        position = key.find(marker)
        if position >= 0:
            extracted[key[position + len(marker):]] = value.detach().cpu()
    return extracted


def main():
    parser = argparse.ArgumentParser(description="导出轻量笔触 embedding 权重")
    parser.add_argument("--ckpt", required=True, help="完整训练 checkpoint")
    parser.add_argument("--output", default="./weights/stroke_embedding.pt", help="轻量权重输出路径")
    args = parser.parse_args()

    try:
        checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.ckpt, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    feature_state = extract_prefixed(state_dict, "stroke_feature_extractor.")
    adapter_state = extract_prefixed(state_dict, "stroke_style_adapter.")
    if not feature_state or not adapter_state:
        raise RuntimeError(
            f"checkpoint 中缺少笔触模块: feature={len(feature_state)} adapter={len(adapter_state)}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stroke_feature_extractor": feature_state,
            "stroke_style_adapter": adapter_state,
            "source_checkpoint": str(Path(args.ckpt).resolve()),
        },
        output_path,
    )
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"exported feature tensors: {len(feature_state)}")
    print(f"exported adapter tensors: {len(adapter_state)}")
    print(f"output: {output_path.resolve()} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
