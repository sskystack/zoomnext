#!/usr/bin/env python3
"""Validate the migrated Jittor RN50_ZoomNeXt model against the PyTorch reference."""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from typing import Dict

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from methods.zoomnext.zoomnext import RN50_ZoomNeXt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Jittor RN50_ZoomNeXt against PyTorch.")
    parser.add_argument("--seed", type=int, default=20260414)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=1)
    parser.add_argument("--mid-dim", type=int, default=64)
    parser.add_argument("--siu-groups", type=int, default=4)
    parser.add_argument("--hmu-groups", type=int, default=6)
    parser.add_argument("--height-l", type=int, default=352)
    parser.add_argument("--width-l", type=int, default=352)
    parser.add_argument("--height-m", type=int, default=352)
    parser.add_argument("--width-m", type=int, default=352)
    parser.add_argument("--height-s", type=int, default=352)
    parser.add_argument("--width-s", type=int, default=352)
    parser.add_argument(
        "--encoder-weight-path",
        type=pathlib.Path,
        default=REPO_ROOT / "pretrained_weights" / "resnet50-timm.pth",
    )
    parser.add_argument("--tol-max", type=float, default=3e-4)
    parser.add_argument("--tol-mean", type=float, default=1e-6)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compare_arrays(name: str, pt_value: torch.Tensor, jt_value: np.ndarray) -> Dict[str, object]:
    pt_arr = pt_value.detach().cpu().numpy().astype(np.float32)
    jt_arr = np.asarray(jt_value, dtype=np.float32)
    abs_err = np.abs(pt_arr - jt_arr)
    return {
        "name": name,
        "shape": list(pt_arr.shape),
        "max_abs_err": float(abs_err.max()),
        "mean_abs_err": float(abs_err.mean()),
    }


def map_torch_state_dict_to_jittor(state_dict: dict) -> dict:
    mapped = {}
    for name, value in state_dict.items():
        if name in {"normalizer.mean", "normalizer.std"}:
            continue

        mapped_name = name
        mapped_name = mapped_name.replace("gate_genator.1.", "gate_conv1.")
        mapped_name = mapped_name.replace("gate_genator.3.", "gate_conv2.")
        mapped_name = mapped_name.replace("fuse.0.", "fuse_diff.")
        mapped_name = mapped_name.replace("fuse.1.", "fuse_conv.")
        for group_id in range(100):
            prefix = f"interact.{group_id}."
            if prefix in mapped_name:
                mapped_name = mapped_name.replace(prefix, f"interact_{group_id}.")
                break
        mapped[mapped_name] = value.detach().cpu().numpy()
    return mapped


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    try:
        import jittor as jt
    except ImportError as exc:
        raise SystemExit(
            "Jittor is not installed. Please install Jittor in the container before running this script."
        ) from exc

    from jittor_impl.models import RN50_ZoomNeXt_JT

    jt.flags.use_cuda = 0

    image_l = np.random.randn(args.batch_size, 3, args.height_l, args.width_l).astype(np.float32)
    image_m = np.random.randn(args.batch_size, 3, args.height_m, args.width_m).astype(np.float32)
    image_s = np.random.randn(args.batch_size, 3, args.height_s, args.width_s).astype(np.float32)

    pt_inputs = {
        "image_l": torch.from_numpy(image_l),
        "image_m": torch.from_numpy(image_m),
        "image_s": torch.from_numpy(image_s),
    }
    jt_inputs = {
        "image_l": jt.array(image_l),
        "image_m": jt.array(image_m),
        "image_s": jt.array(image_s),
    }

    pt_model = RN50_ZoomNeXt(
        pretrained=False,
        num_frames=args.num_frames,
        input_norm=True,
        mid_dim=args.mid_dim,
        siu_groups=args.siu_groups,
        hmu_groups=args.hmu_groups,
    )
    if args.encoder_weight_path.is_file():
        encoder_state = torch.load(args.encoder_weight_path, map_location="cpu")
        pt_model.encoder.load_state_dict(encoder_state, strict=False)
    pt_model.eval()

    jt_model = RN50_ZoomNeXt_JT(
        pretrained=False,
        num_frames=args.num_frames,
        input_norm=True,
        mid_dim=args.mid_dim,
        siu_groups=args.siu_groups,
        hmu_groups=args.hmu_groups,
    )
    jt_model.eval()
    jt_model.load_state_dict(map_torch_state_dict_to_jittor(pt_model.state_dict()))

    with torch.no_grad():
        pt_output = pt_model(pt_inputs)
    jt_output = jt_model(jt_inputs).numpy()

    report = compare_arrays("RN50_ZoomNeXt", pt_output, jt_output)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if report["max_abs_err"] > args.tol_max or report["mean_abs_err"] > args.tol_mean:
        print(
            "\nValidation failed: "
            f"max_abs_err={report['max_abs_err']:.6e}, mean_abs_err={report['mean_abs_err']:.6e}"
        )
        return 1

    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
