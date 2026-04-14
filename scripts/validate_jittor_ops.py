#!/usr/bin/env python3
"""Validate the migrated Jittor base operators against the PyTorch reference.

This script is intended to run inside an Ubuntu container where both
PyTorch and Jittor are available. It only validates step-2 operators.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from typing import Dict, Tuple

import numpy as np
import torch

# Make the repo root importable even when this script is executed via
# `python scripts/validate_jittor_ops.py`.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from methods.zoomnext.ops import LayerNorm2d as TorchLayerNorm2d
from methods.zoomnext.ops import PixelNormalizer as TorchPixelNormalizer
from methods.zoomnext.ops import resize_to as torch_resize_to
from methods.zoomnext.ops import rescale_2x as torch_rescale_2x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Jittor base ops against PyTorch.")
    parser.add_argument("--seed", type=int, default=20260414)
    parser.add_argument("--tol-max", type=float, default=1e-5)
    parser.add_argument("--tol-mean", type=float, default=1e-6)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def compare_arrays(name: str, pt_value, jt_value) -> Dict[str, float]:
    pt_arr = to_numpy(pt_value).astype(np.float32)
    jt_arr = to_numpy(jt_value).astype(np.float32)
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

    try:
        import jittor as jt
    except ImportError as exc:
        raise SystemExit(
            "Jittor is not installed. Please install Jittor in the container before running this script."
        ) from exc

    from jittor_impl.models.ops_jt import LayerNorm2d as JittorLayerNorm2d
    from jittor_impl.models.ops_jt import PixelNormalizer as JittorPixelNormalizer
    from jittor_impl.models.ops_jt import resize_to as jittor_resize_to
    from jittor_impl.models.ops_jt import rescale_2x as jittor_rescale_2x

    jt.flags.use_cuda = 0

    x_np = np.random.randn(2, 3, 15, 19).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_jt = jt.array(x_np)

    reports = []

    reports.append(
        compare_arrays(
            "resize_to",
            torch_resize_to(x_pt, tgt_hw=(11, 13)),
            jittor_resize_to(x_jt, tgt_hw=(11, 13)).numpy(),
        )
    )
    reports.append(
        compare_arrays(
            "rescale_2x",
            torch_rescale_2x(x_pt, scale_factor=2),
            jittor_rescale_2x(x_jt, scale_factor=2).numpy(),
        )
    )

    pt_norm = TorchPixelNormalizer()
    jt_norm = JittorPixelNormalizer()
    reports.append(compare_arrays("PixelNormalizer", pt_norm(x_pt), jt_norm(x_jt).numpy()))

    pt_ln = TorchLayerNorm2d(3)
    jt_ln = JittorLayerNorm2d(3)
    weight = np.array([1.1, 0.9, -0.7], dtype=np.float32)
    bias = np.array([0.2, -0.3, 0.5], dtype=np.float32)
    with torch.no_grad():
        pt_ln.weight.copy_(torch.from_numpy(weight))
        pt_ln.bias.copy_(torch.from_numpy(bias))
    jt_ln.load_state_dict({"weight": weight, "bias": bias})
    reports.append(compare_arrays("LayerNorm2d", pt_ln(x_pt), jt_ln(x_jt).numpy()))

    print(json.dumps(reports, indent=2, ensure_ascii=False))

    failed = [
        item
        for item in reports
        if item["max_abs_err"] > args.tol_max or item["mean_abs_err"] > args.tol_mean
    ]
    if failed:
        print("\nValidation failed for:")
        for item in failed:
            print(
                f"- {item['name']}: max_abs_err={item['max_abs_err']:.6e}, "
                f"mean_abs_err={item['mean_abs_err']:.6e}"
            )
        return 1

    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
