#!/usr/bin/env python3
"""Validate the Jittor RN50_ZoomNeXt training path against the PyTorch reference."""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
import sys
from typing import Dict

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from methods.zoomnext.zoomnext import RN50_ZoomNeXt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Jittor RN50_ZoomNeXt training path against PyTorch.")
    parser.add_argument("--seed", type=int, default=20260414)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=1)
    parser.add_argument("--mid-dim", type=int, default=64)
    parser.add_argument("--siu-groups", type=int, default=4)
    parser.add_argument("--hmu-groups", type=int, default=6)
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--width", type=int, default=352)
    parser.add_argument("--iter-percentage", type=float, default=0.37)
    parser.add_argument(
        "--encoder-weight-path",
        type=pathlib.Path,
        default=REPO_ROOT / "pretrained_weights" / "resnet50-timm.pth",
    )
    parser.add_argument("--tol-sal-max", type=float, default=5e-4)
    parser.add_argument("--tol-sal-mean", type=float, default=1e-5)
    parser.add_argument("--tol-loss", type=float, default=1e-4)
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


def parameter_count(params) -> int:
    return sum(int(np.prod(tuple(param.shape))) for param in params)


def summarize_groups(groups: dict) -> Dict[str, Dict[str, int]]:
    return {
        group_name: {
            "params": len(group_params),
            "elements": parameter_count(group_params),
        }
        for group_name, group_params in groups.items()
    }


def freeze_torch_encoder_bn_stats(model, freeze_affine: bool = False) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
            if freeze_affine:
                module.requires_grad_(False)


def parse_loss_str(loss_str: str) -> Dict[str, float]:
    parsed = {}
    for key, value in re.findall(r"([A-Za-z0-9_\.]+): ([0-9.]+)", loss_str):
        parsed[key] = float(value)
    return parsed


def compare_loss_strings(pt_loss_str: str, jt_loss_str: str, tol: float) -> bool:
    pt_items = parse_loss_str(pt_loss_str)
    jt_items = parse_loss_str(jt_loss_str)
    if pt_items.keys() != jt_items.keys():
        return False
    for key in pt_items:
        if abs(pt_items[key] - jt_items[key]) > tol:
            return False
    return True


def map_torch_state_dict_to_jittor(state_dict: dict) -> dict:
    mapped = {}
    for name, value in state_dict.items():
        if name in {"normalizer.mean", "normalizer.std"}:
            continue

        mapped_name = name
        if mapped_name.startswith("hmu_"):
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
    from jittor_impl.models.zoomnext_jt import frozen_bn_stats_jt

    jt.flags.use_cuda = 0

    image_l = np.random.randn(args.batch_size, 3, args.height, args.width).astype(np.float32)
    image_m = np.random.randn(args.batch_size, 3, args.height, args.width).astype(np.float32)
    image_s = np.random.randn(args.batch_size, 3, args.height, args.width).astype(np.float32)
    mask = np.random.rand(args.batch_size, 1, args.height, args.width).astype(np.float32)

    pt_inputs = {
        "image_l": torch.from_numpy(image_l),
        "image_m": torch.from_numpy(image_m),
        "image_s": torch.from_numpy(image_s),
        "mask": torch.from_numpy(mask),
    }
    jt_inputs = {
        "image_l": jt.array(image_l),
        "image_m": jt.array(image_m),
        "image_s": jt.array(image_s),
        "mask": jt.array(mask),
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
    pt_model.train()

    jt_model = RN50_ZoomNeXt_JT(
        pretrained=False,
        num_frames=args.num_frames,
        input_norm=True,
        mid_dim=args.mid_dim,
        siu_groups=args.siu_groups,
        hmu_groups=args.hmu_groups,
    )
    jt_model.train()
    jt_model.load_state_dict(map_torch_state_dict_to_jittor(pt_model.state_dict()))

    pt_groups = summarize_groups(pt_model.get_grouped_params())
    jt_groups = summarize_groups(jt_model.get_grouped_params())

    freeze_torch_encoder_bn_stats(pt_model.encoder, freeze_affine=True)
    frozen_bn_stats_jt(jt_model.encoder, freeze_affine=True)

    pt_output = pt_model(pt_inputs, iter_percentage=args.iter_percentage)
    jt_output = jt_model(jt_inputs, iter_percentage=args.iter_percentage)

    sal_report = compare_arrays("sal", pt_output["vis"]["sal"], jt_output["vis"]["sal"].numpy())
    loss_abs_err = abs(float(pt_output["loss"].detach().cpu().item()) - float(jt_output["loss"].numpy()))
    loss_str_exact_match = pt_output["loss_str"] == jt_output["loss_str"]
    loss_str_value_match = compare_loss_strings(pt_output["loss_str"], jt_output["loss_str"], tol=1e-4)

    report = {
        "saliency": sal_report,
        "loss_abs_err": loss_abs_err,
        "loss_str_exact_match": loss_str_exact_match,
        "loss_str_value_match": loss_str_value_match,
        "pytorch_loss_str": pt_output["loss_str"],
        "jittor_loss_str": jt_output["loss_str"],
        "param_groups_match": pt_groups == jt_groups,
        "pytorch_param_groups": pt_groups,
        "jittor_param_groups": jt_groups,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if sal_report["max_abs_err"] > args.tol_sal_max or sal_report["mean_abs_err"] > args.tol_sal_mean:
        print(
            "\nValidation failed: "
            f"sal_max_abs_err={sal_report['max_abs_err']:.6e}, sal_mean_abs_err={sal_report['mean_abs_err']:.6e}"
        )
        return 1
    if loss_abs_err > args.tol_loss:
        print(f"\nValidation failed: loss_abs_err={loss_abs_err:.6e}")
        return 1
    if not loss_str_value_match:
        print("\nValidation failed: loss_str value mismatch")
        return 1
    if pt_groups != jt_groups:
        print("\nValidation failed: grouped params mismatch")
        return 1

    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
