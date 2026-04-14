#!/usr/bin/env python3
"""Validate the migrated Jittor ResNet50 backbone against the PyTorch reference."""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from typing import Dict, List

import numpy as np
import timm
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Jittor ResNet50 backbone against PyTorch/timm.")
    parser.add_argument("--seed", type=int, default=20260414)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--width", type=int, default=352)
    parser.add_argument(
        "--weight-path",
        type=pathlib.Path,
        default=REPO_ROOT / "pretrained_weights" / "resnet50-timm.pth",
    )
    parser.add_argument("--tol-max", type=float, default=2e-4)
    parser.add_argument("--tol-mean", type=float, default=1e-6)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compare_feature(name: str, pt_value: torch.Tensor, jt_value: np.ndarray) -> Dict[str, object]:
    pt_arr = pt_value.detach().cpu().numpy().astype(np.float32)
    jt_arr = np.asarray(jt_value, dtype=np.float32)
    abs_err = np.abs(pt_arr - jt_arr)
    return {
        "name": name,
        "shape": list(pt_arr.shape),
        "max_abs_err": float(abs_err.max()),
        "mean_abs_err": float(abs_err.mean()),
    }


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    if not args.weight_path.is_file():
        raise SystemExit(f"Cannot find pretrained weights: {args.weight_path}")

    try:
        import jittor as jt
    except ImportError as exc:
        raise SystemExit(
            "Jittor is not installed. Please install Jittor in the container before running this script."
        ) from exc

    from jittor_impl.models.backbone import build_resnet50

    jt.flags.use_cuda = 0

    x_np = np.random.randn(args.batch_size, 3, args.height, args.width).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_jt = jt.array(x_np)

    pt_model = timm.create_model("resnet50", features_only=True, out_indices=range(5), pretrained=False)
    pt_state = torch.load(args.weight_path, map_location="cpu")
    pt_model.load_state_dict(pt_state, strict=False)
    pt_model.eval()

    jt_model = build_resnet50(pretrained=True, weight_path=str(args.weight_path))
    jt_model.eval()

    with torch.no_grad():
        pt_outputs = pt_model(x_pt)
    jt_outputs = jt_model(x_jt)

    reports: List[Dict[str, object]] = []
    max_err = 0.0
    mean_err = 0.0
    for idx, (pt_value, jt_value) in enumerate(zip(pt_outputs, jt_outputs), start=1):
        report = compare_feature(f"c{idx}", pt_value, jt_value.numpy())
        reports.append(report)
        max_err = max(max_err, report["max_abs_err"])
        mean_err = max(mean_err, report["mean_abs_err"])

    print(json.dumps(reports, indent=2, ensure_ascii=False))

    if max_err > args.tol_max or mean_err > args.tol_mean:
        print(f"\nValidation failed: max_abs_err={max_err:.6e}, max_mean_abs_err={mean_err:.6e}")
        return 1

    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
