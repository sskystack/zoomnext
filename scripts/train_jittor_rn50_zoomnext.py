#!/usr/bin/env python3
"""Train the Jittor RN50_ZoomNeXt model with dataset configs."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import yaml
from mmengine import Config
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jittor_impl.models import RN50_ZoomNeXt_JT
from jittor_impl.models.zoomnext_jt import frozen_bn_stats_jt
from jittor_impl.pipeline import ScalerJT, SchedulerJT
from utils import io, ops, py_utils


def emit_progress(progress, *, epoch: int, curr_iter: int, stage: str, extra: str = "") -> None:
    progress.set_description(f"[JT-TRAIN][E{epoch}]")
    suffix = f"iter={curr_iter} stage={stage}"
    if extra:
        suffix = f"{suffix} {extra}"
    progress.set_postfix_str(suffix)
    progress.refresh()


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def append_text_log(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(text + "\n")


def prepare_run_dirs(args: argparse.Namespace, cfg) -> dict[str, Path]:
    exp_name = py_utils.construct_exp_name(model_name="RN50_ZoomNeXt_JT", cfg=cfg)
    path_cfg = py_utils.construct_path(output_dir=args.output_dir, exp_name=exp_name)
    py_utils.pre_mkdir(path_cfg)

    run_dir = Path(path_cfg["pth_log"])
    checkpoints_dir = Path(path_cfg["pth"])
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    config_copy_path = Path(path_cfg["cfg_copy"])
    trainer_copy_path = Path(path_cfg["trainer_copy"])
    log_path = Path(path_cfg["log"])
    iter_log_path = run_dir / "train_iter.jsonl"
    ckpt_log_path = run_dir / "checkpoint_log.jsonl"
    run_meta_path = run_dir / "run_meta.json"

    shutil.copy2(args.config, config_copy_path)
    shutil.copy2(Path(__file__).resolve(), trainer_copy_path)

    run_meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(Path(args.config).resolve()),
        "data_cfg": str(Path(args.data_cfg).resolve()),
        "output_dir": str(Path(args.output_dir).resolve()),
        "train_datasets": list(cfg.train.data.names),
        "effective_batch_size": int(cfg.train.batch_size),
        "effective_num_epochs": int(cfg.train.num_epochs),
        "effective_lr": float(cfg.train.lr),
        "effective_grad_acc_step": int(cfg.train.grad_acc_step),
        "effective_use_amp": bool(cfg.train.use_amp),
        "effective_sche_usebatch": bool(cfg.train.sche_usebatch),
        "effective_scheduler": cfg.train.scheduler,
        "pretrained": args.pretrained,
        "encoder_weight_path": args.encoder_weight_path,
        "resume_from": args.resume_from,
        "max_iters": args.max_iters,
        "save_every": args.save_every,
        "seed": args.seed,
        "use_cuda": args.use_cuda,
    }
    run_meta_path.write_text(json.dumps(run_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    append_text_log(log_path, f"=== Jittor training run {run_meta['created_at']} ===")
    append_text_log(log_path, json.dumps(run_meta, ensure_ascii=False))

    return {
        "run_dir": run_dir,
        "checkpoints_dir": checkpoints_dir,
        "config_copy": config_copy_path,
        "trainer_copy": trainer_copy_path,
        "log": log_path,
        "iter_log": iter_log_path,
        "ckpt_log": ckpt_log_path,
        "run_meta": run_meta_path,
    }


class ImageTrainDatasetJT:
    def __init__(self, dataset_infos: dict, shape: dict):
        self.shape = shape
        self.total_data_paths = []
        for dataset_name, dataset_info in dataset_infos.items():
            image_path = os.path.join(dataset_info["root"], dataset_info["image"]["path"])
            image_suffix = dataset_info["image"]["suffix"]
            mask_path = os.path.join(dataset_info["root"], dataset_info["mask"]["path"])
            mask_suffix = dataset_info["mask"]["suffix"]

            image_names = [p[: -len(image_suffix)] for p in sorted(os.listdir(image_path)) if p.endswith(image_suffix)]
            mask_names = [p[: -len(mask_suffix)] for p in sorted(os.listdir(mask_path)) if p.endswith(mask_suffix)]
            valid_names = sorted(set(image_names).intersection(mask_names))
            data_paths = [
                (os.path.join(image_path, n) + image_suffix, os.path.join(mask_path, n) + mask_suffix)
                for n in valid_names
            ]
            print(json.dumps({"dataset": dataset_name, "length": len(data_paths)}, ensure_ascii=False))
            self.total_data_paths.extend(data_paths)

        self.trains = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=90, p=0.5, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REPLICATE),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.5),
            ]
        )

    def __len__(self):
        return len(self.total_data_paths)

    def __getitem__(self, index):
        image_path, mask_path = self.total_data_paths[index]
        image = io.read_color_array(image_path)
        mask = io.read_gray_array(mask_path, thr=0)
        if image.shape[:2] != mask.shape:
            h, w = mask.shape
            image = ops.resize(image, height=h, width=w)

        transformed = self.trains(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]

        base_h = self.shape["h"]
        base_w = self.shape["w"]
        images = ops.ms_resize(image, scales=(0.5, 1.0, 1.5), base_h=base_h, base_w=base_w)

        image_s = images[0].astype(np.float32) / 255.0
        image_m = images[1].astype(np.float32) / 255.0
        image_l = images[2].astype(np.float32) / 255.0
        image_s = np.transpose(image_s, (2, 0, 1))
        image_m = np.transpose(image_m, (2, 0, 1))
        image_l = np.transpose(image_l, (2, 0, 1))

        mask = ops.resize(mask, height=base_h, width=base_w).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        return {
            "image_s": image_s,
            "image_m": image_m,
            "image_l": image_l,
            "mask": mask,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Jittor RN50_ZoomNeXt training script")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--data-cfg", required=True, type=str)
    parser.add_argument("--output-dir", type=str, default="outputs_jittor")
    parser.add_argument("--train-datasets", nargs="+", type=str)
    parser.add_argument("--num-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--encoder-weight-path", type=str, default="pretrained_weights/resnet50-timm.pth")
    parser.add_argument("--resume-from", type=str)
    parser.add_argument("--max-iters", type=int)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=112358)
    parser.add_argument("--use-cuda", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_cfg(args: argparse.Namespace):
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(
        {
            "data_cfg": args.data_cfg,
            "output_dir": args.output_dir,
            "pretrained": args.pretrained,
        }
    )
    with open(args.data_cfg, mode="r", encoding="utf-8") as f:
        cfg.dataset_infos = yaml.safe_load(f)
    if args.train_datasets:
        cfg.train.data.names = list(args.train_datasets)
    if args.num_epochs is not None:
        cfg.train.num_epochs = int(args.num_epochs)
    if args.batch_size is not None:
        cfg.train.batch_size = int(args.batch_size)
    if args.lr is not None:
        cfg.train.lr = float(args.lr)
    return cfg


def batch_indices(num_samples: int, batch_size: int, *, seed: int, drop_last: bool = True):
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    for start in range(0, num_samples, batch_size):
        batch = indices[start : start + batch_size]
        if len(batch) < batch_size and drop_last:
            continue
        yield batch.tolist()


def collate_samples(samples: list[dict]) -> dict[str, np.ndarray]:
    return {key: np.stack([sample[key] for sample in samples], axis=0).astype(np.float32) for key in samples[0]}


def filter_trainable(params):
    return [param for param in params if getattr(param, "requires_grad", True)]


def group_params_jt(model, group_mode: str, initial_lr: float, optim_cfg: dict):
    if group_mode == "all":
        return [param for _, param in model.named_parameters() if getattr(param, "requires_grad", True)]
    if group_mode == "finetune":
        params_groups = model.get_grouped_params()
        return [
            {
                "params": filter_trainable(params_groups["pretrained"]),
                "lr": optim_cfg.get("diff_factor", 0.1) * initial_lr,
            },
            {
                "params": filter_trainable(params_groups["retrained"]),
                "lr": initial_lr,
            },
        ]
    raise NotImplementedError(f"Unsupported group_mode for current Jittor script: {group_mode}")


def construct_optimizer_jt(jt, model, initial_lr: float, mode: str, group_mode: str, optim_cfg: dict):
    params = group_params_jt(model, group_mode=group_mode, initial_lr=initial_lr, optim_cfg=optim_cfg)
    if mode == "adam":
        optimizer = jt.optim.Adam(
            params=params,
            lr=initial_lr,
            betas=tuple(optim_cfg.get("betas", (0.9, 0.999))),
            weight_decay=optim_cfg.get("weight_decay", 0),
        )
    elif mode == "adamw":
        optimizer = jt.optim.AdamW(
            params=params,
            lr=initial_lr,
            betas=tuple(optim_cfg.get("betas", (0.9, 0.999))),
            weight_decay=optim_cfg.get("weight_decay", 0),
        )
    elif mode == "sgd":
        optimizer = jt.optim.SGD(
            params=params,
            lr=initial_lr,
            momentum=optim_cfg.get("momentum", 0),
            weight_decay=optim_cfg.get("weight_decay", 0),
            nesterov=optim_cfg.get("nesterov", False),
        )
    else:
        raise NotImplementedError(mode)
    return optimizer


def lr_groups(optimizer) -> list[float]:
    return [group.get("lr", optimizer.lr) for group in optimizer.param_groups]


def lr_string(optimizer) -> str:
    return ",".join(f"{lr:.3e}" for lr in lr_groups(optimizer))


def maybe_freeze_encoder(model, freeze_encoder: bool):
    if not freeze_encoder:
        return
    for _, param in model.encoder.named_parameters():
        param.requires_grad = False


def maybe_freeze_bn_stats(model, freeze_status: bool, freeze_affine: bool):
    if freeze_status:
        frozen_bn_stats_jt(model.encoder, freeze_affine=freeze_affine)


def save_checkpoint(jt, model, save_dir: Path, step: int):
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"step_{step:06d}.pkl"
    jt.save(model.state_dict(), str(save_path))
    return save_path


def construct_scheduler_jt(optimizer, total_iters: int, epoch_length: int, train_cfg):
    scheduler = SchedulerJT(
        optimizer=optimizer,
        num_iters=total_iters,
        epoch_length=epoch_length,
        scheduler_cfg=train_cfg.scheduler,
        step_by_batch=train_cfg.sche_usebatch,
    )
    scheduler.record_lrs(param_groups=optimizer.param_groups)
    return scheduler


def construct_scaler_jt(jt, optimizer, train_cfg):
    optimizer_cfg = train_cfg.optimizer
    return ScalerJT(
        jt=jt,
        optimizer=optimizer,
        use_fp16=bool(train_cfg.use_amp),
        set_to_none=bool(optimizer_cfg.get("set_to_none", False)),
        clip_grad=bool(optimizer_cfg.get("clip_grad", False)),
        clip_mode=optimizer_cfg.get("clip_mode"),
        clip_cfg=optimizer_cfg.get("clip_cfg"),
        amp_level=int(train_cfg.get("amp_level", 5)),
    )


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    cfg = load_cfg(args)

    import jittor as jt

    jt.flags.use_cuda = 1 if args.use_cuda else 0
    run_paths = prepare_run_dirs(args, cfg)

    train_names = list(cfg.train.data.names)
    dataset = ImageTrainDatasetJT(
        dataset_infos={data_name: cfg.dataset_infos[data_name] for data_name in train_names},
        shape=cfg.train.data.shape,
    )
    if len(dataset) == 0:
        raise SystemExit("The training dataset is empty. Please check --data-cfg and dataset paths.")

    model = RN50_ZoomNeXt_JT(
        pretrained=args.pretrained,
        num_frames=1,
        input_norm=True,
        mid_dim=64,
        siu_groups=4,
        hmu_groups=6,
        weight_path=args.encoder_weight_path,
    )
    if args.resume_from:
        model.load_state_dict(jt.load(args.resume_from))
    model.train()

    maybe_freeze_encoder(model, cfg.train.bn.freeze_encoder)

    optimizer = construct_optimizer_jt(
        jt=jt,
        model=model,
        initial_lr=cfg.train.lr,
        mode=cfg.train.optimizer.mode,
        group_mode=cfg.train.optimizer.group_mode,
        optim_cfg=cfg.train.optimizer.cfg,
    )

    checkpoint_dir = run_paths["checkpoints_dir"]

    batch_size = int(cfg.train.batch_size)
    num_epochs = int(cfg.train.num_epochs)
    epoch_length = max(1, len(dataset) // batch_size)
    total_iters = args.max_iters or (num_epochs * epoch_length)
    if args.max_iters is not None:
        num_epochs = max(num_epochs, (total_iters + epoch_length - 1) // epoch_length)
    scheduler = construct_scheduler_jt(optimizer=optimizer, total_iters=total_iters, epoch_length=epoch_length, train_cfg=cfg.train)
    scheduler.plot_lr_coef_curve(save_path=str(run_paths["run_dir"]))
    scaler = construct_scaler_jt(jt=jt, optimizer=optimizer, train_cfg=cfg.train)
    grad_acc_step = max(1, int(cfg.train.grad_acc_step))
    curr_iter = 0

    start_time = time.perf_counter()
    progress = tqdm(total=total_iters, desc="[JT-TRAIN]", ncols=100, mininterval=0.0, file=sys.stdout)
    append_text_log(run_paths["log"], str(scheduler))
    optimizer.zero_grad()
    for epoch in range(num_epochs):
        model.train()
        maybe_freeze_bn_stats(model, cfg.train.bn.freeze_status, cfg.train.bn.freeze_affine)

        for batch_ids in batch_indices(len(dataset), batch_size, seed=args.seed + epoch, drop_last=True):
            emit_progress(progress, epoch=epoch, curr_iter=curr_iter, stage="load")
            samples = [dataset[idx] for idx in batch_ids]
            np_batch = collate_samples(samples)
            jt_batch = {key: jt.array(value) for key, value in np_batch.items()}

            iter_percentage = curr_iter / max(total_iters - 1, 1)
            scheduler.step(curr_iter)
            emit_progress(progress, epoch=epoch, curr_iter=curr_iter, stage="forward")
            with scaler.autocast():
                outputs = model(data=jt_batch, iter_percentage=iter_percentage)
            loss = outputs["loss"]
            loss = loss / grad_acc_step
            emit_progress(progress, epoch=epoch, curr_iter=curr_iter, stage="backward")
            scaler.calculate_grad(loss)
            if (curr_iter + 1) % grad_acc_step == 0 or curr_iter == total_iters - 1:
                scaler.update_grad()

            item = {
                "iter": curr_iter,
                "epoch": epoch,
                "lr": lr_groups(optimizer),
                "lr_string": lr_string(optimizer),
                "loss": float(loss.numpy()),
                "loss_str": outputs["loss_str"],
                "shape": list(np_batch["mask"].shape),
            }
            progress.update(1)
            emit_progress(
                progress,
                epoch=epoch,
                curr_iter=curr_iter,
                stage="done",
                extra=f"loss={item['loss']:.5f} lr={item['lr_string']}",
            )
            progress.write(json.dumps(item, ensure_ascii=False))
            append_jsonl(run_paths["iter_log"], item)
            append_text_log(run_paths["log"], json.dumps(item, ensure_ascii=False))

            curr_iter += 1
            if curr_iter % args.save_every == 0 or curr_iter == total_iters:
                emit_progress(progress, epoch=epoch, curr_iter=curr_iter, stage="save")
                save_path = save_checkpoint(jt, model, checkpoint_dir, curr_iter)
                ckpt_item = {"checkpoint": str(save_path), "iter": curr_iter, "epoch": epoch}
                progress.write(json.dumps(ckpt_item, ensure_ascii=False))
                append_jsonl(run_paths["ckpt_log"], ckpt_item)
                append_text_log(run_paths["log"], json.dumps(ckpt_item, ensure_ascii=False))

            if curr_iter >= total_iters:
                break
        if curr_iter >= total_iters:
            break

    progress.close()
    final_path = save_checkpoint(jt, model, checkpoint_dir, curr_iter)
    elapsed = time.perf_counter() - start_time
    final_item = {"final_checkpoint": str(final_path), "elapsed_sec": elapsed, "iters": curr_iter}
    print(json.dumps(final_item, ensure_ascii=False))
    append_jsonl(run_paths["ckpt_log"], final_item)
    append_text_log(run_paths["log"], json.dumps(final_item, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
