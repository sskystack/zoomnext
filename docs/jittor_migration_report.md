# ZoomNeXt 到 Jittor 迁移实验报告

## 1. 本轮工作的目标

本轮工作对应迁移计划中的前两步：

1. 先建立 Jittor 迁移目录骨架，明确后续代码落点
2. 先完成最基础的一层算子迁移，并提供 PyTorch/Jittor 数值对比脚本

当前这一轮的范围还不包括：

- `SimpleASPP`
- `MHSIU`
- `DifferenceAwareOps`
- `RGPU`
- `ResNet50` backbone
- `RN50_ZoomNeXt` 整体前向

因此，本轮结论只能说明“基础算子层已经开始建立，并且当前实现通过了第一轮数值验证”，还不能说明整个 `RN50_ZoomNeXt` 已经完成等价迁移。

## 2. 本轮修改内容概览

本轮新增或修改的迁移相关文件如下：

- `jittor_impl/__init__.py`
- `jittor_impl/models/__init__.py`
- `jittor_impl/models/backbone/__init__.py`
- `jittor_impl/models/ops_jt.py`
- `jittor_impl/models/layers_jt.py`
- `jittor_impl/models/backbone/resnet50_jt.py`
- `jittor_impl/models/zoomnext_jt.py`
- `scripts/validate_jittor_ops.py`

其中：

- `ops_jt.py` 是本轮真正实现完成并做了验证的部分
- `layers_jt.py`、`resnet50_jt.py`、`zoomnext_jt.py` 目前只是明确迁移入口的占位文件
- `validate_jittor_ops.py` 是本轮验证脚本

## 3. 代码变更与解释

下面按照“文件 -> 完整代码 -> 解释”的形式记录本轮全部迁移相关代码。

### 3.1 `jittor_impl/__init__.py`

完整代码：

```python
"""Jittor migration workspace for ZoomNeXt."""
```

解释：

这个文件的作用很简单，是为了把 `jittor_impl/` 显式声明为 Python 包。这样后续脚本才能通过 `import jittor_impl...` 的方式稳定导入迁移代码。

### 3.2 `jittor_impl/models/__init__.py`

完整代码：

```python
"""Model modules for the Jittor ZoomNeXt migration."""
```

解释：

这个文件把 `jittor_impl/models/` 声明为包，方便后续模型层、骨干网络层、整网层的导入组织。

### 3.3 `jittor_impl/models/backbone/__init__.py`

完整代码：

```python
"""Backbone modules for the Jittor ZoomNeXt migration."""
```

解释：

这个文件把 `jittor_impl/models/backbone/` 声明为包，为后续 `resnet50_jt.py`、`pvtv2_jt.py`、`efficientnet_jt.py` 提供标准落点。

### 3.4 `jittor_impl/models/ops_jt.py`

完整代码：

```python
"""Jittor counterparts for the common ZoomNeXt operators.

This file intentionally mirrors the structure of
`methods/zoomnext/ops.py` and `utils/ops/tensor_ops.py` so we can
compare small operator blocks before migrating the whole model.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import jittor as jt
from jittor import nn


def _to_2tuple(value) -> Tuple[int, int]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"Expected a 2-tuple/list, got {value!r}")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def rescale_2x(x: jt.Var, scale_factor=2) -> jt.Var:
    return nn.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)


def resize_to(x: jt.Var, tgt_hw: tuple) -> jt.Var:
    return nn.interpolate(x, size=tgt_hw, mode="bilinear", align_corners=False)


def global_avgpool(x: jt.Var) -> jt.Var:
    return x.mean(dims=(-1, -2), keepdims=True)


class _LeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 0.1):
        super().__init__()
        self.negative_slope = negative_slope

    def execute(self, x: jt.Var) -> jt.Var:
        return nn.leaky_relu(x, scale=self.negative_slope)


class _GELU(nn.Module):
    def execute(self, x: jt.Var) -> jt.Var:
        return 0.5 * x * (1.0 + jt.erf(x / math.sqrt(2.0)))


def _get_act_fn(act_name: str, inplace: bool = True) -> nn.Module:
    del inplace  # Jittor activations do not use PyTorch-style inplace flags here.
    if act_name == "relu":
        return nn.ReLU()
    if act_name == "leaklyrelu":
        return _LeakyReLU(negative_slope=0.1)
    if act_name == "gelu":
        return _GELU()
    if act_name == "sigmoid":
        return nn.Sigmoid()
    raise NotImplementedError(act_name)


class ConvBN(nn.Module):
    def __init__(self, in_dim, out_dim, k, s=1, p=0, d=1, g=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=k,
            stride=_to_2tuple(s),
            padding=_to_2tuple(p),
            dilation=_to_2tuple(d),
            groups=g,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_dim)

    def execute(self, x: jt.Var) -> jt.Var:
        return self.bn(self.conv(x))


class CBR(nn.Module):
    def __init__(self, in_dim, out_dim, k, s=1, p=0, d=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=k,
            stride=_to_2tuple(s),
            padding=_to_2tuple(p),
            dilation=_to_2tuple(d),
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()

    def execute(self, x: jt.Var) -> jt.Var:
        return self.relu(self.bn(self.conv(x)))


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        act_name="relu",
        is_transposed=False,
    ):
        super().__init__()
        conv_cls = nn.ConvTranspose if is_transposed else nn.Conv2d
        self.append(
            conv_cls(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=_to_2tuple(stride),
                padding=_to_2tuple(padding),
                dilation=_to_2tuple(dilation),
                groups=groups,
                bias=bias,
            )
        )
        self.append(nn.BatchNorm2d(out_planes))
        if act_name is not None:
            self.append(_get_act_fn(act_name=act_name))


class ConvGNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        gn_groups=8,
        bias=False,
        act_name="relu",
        inplace=True,
    ):
        super().__init__()
        self.append(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=_to_2tuple(stride),
                padding=_to_2tuple(padding),
                dilation=_to_2tuple(dilation),
                groups=groups,
                bias=bias,
            )
        )
        self.append(nn.GroupNorm(num_groups=gn_groups, num_channels=out_planes))
        if act_name is not None:
            self.append(_get_act_fn(act_name=act_name, inplace=inplace))


class PixelNormalizer(nn.Module):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__()
        # Leading underscores keep these Vars from being treated as trainable parameters in Jittor.
        self._mean = jt.array(mean).reshape((3, 1, 1)).float32()
        self._std = jt.array(std).reshape((3, 1, 1)).float32()

    @property
    def mean(self) -> jt.Var:
        return self._mean

    @property
    def std(self) -> jt.Var:
        return self._std

    def __repr__(self) -> str:
        mean = [float(x) for x in self.mean.reshape((-1,)).numpy().tolist()]
        std = [float(x) for x in self.std.reshape((-1,)).numpy().tolist()]
        return f"{self.__class__.__name__}(mean={mean}, std={std})"

    def execute(self, x: jt.Var) -> jt.Var:
        x = x - self.mean
        x = x / self.std
        return x


class LayerNorm2d(nn.Module):
    """A channel-wise LayerNorm that mirrors the custom PyTorch implementation."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = jt.ones((num_channels,), dtype=jt.float32)
        self.bias = jt.zeros((num_channels,), dtype=jt.float32)
        self.eps = eps

    def execute(self, x: jt.Var) -> jt.Var:
        u = x.mean(dim=1, keepdims=True)
        s = ((x - u) ** 2).mean(dim=1, keepdims=True)
        x = (x - u) / jt.sqrt(s + self.eps)
        x = self.weight.reshape((1, -1, 1, 1)) * x + self.bias.reshape((1, -1, 1, 1))
        return x
```

解释：

这个文件是本轮真正完成迁移和验证的核心。

#### `_to_2tuple`

作用：

- 把整数参数转换成二元组
- 对齐 PyTorch 中 `to_2tuple` 的行为

为什么需要：

- `Conv2d`、`GroupNorm`、插值等接口经常既接受整数也接受二元组
- 提前统一成 `(h, w)` 能减少 Jittor 和 PyTorch 接口差异

#### `rescale_2x` 与 `resize_to`

作用：

- 对应原仓库中的双线性插值接口

关键点：

- 显式指定 `mode="bilinear"`
- 显式指定 `align_corners=False`

这是必须保留的，因为插值行为如果不一致，后续 decoder 和多尺度融合就会出现数值漂移。

#### `global_avgpool`

作用：

- 对应原仓库对空间维 `H, W` 的全局平均池化

关键点：

- 保留 `keepdims=True`

#### `_LeakyReLU` 与 `_GELU`

作用：

- 封装 Jittor 版本激活函数
- 对齐原仓库 `act_name` 分发方式

为什么这样写：

- 原仓库是通过字符串来构建激活函数
- 迁移时保留这个设计，后续复用 `ConvBNReLU`、`ConvGNReLU` 更方便

#### `ConvBN`、`CBR`、`ConvBNReLU`、`ConvGNReLU`

作用：

- 对应原仓库最常用的卷积模块组合

为什么重要：

- 后续 `SimpleASPP`、`MHSIU`、`RGPU` 全部依赖这些模块
- 如果这里行为不一致，后续所有层都会一起偏掉

#### `PixelNormalizer`

作用：

- 对应原仓库的输入归一化模块

为什么这样实现：

- 原仓库使用 `register_buffer` 存 `mean/std`
- Jittor 这里先用 `_mean/_std` 保存常量，避免它们被当成训练参数
- 对外仍保留 `mean/std` 属性，方便后续日志输出和兼容原接口

#### `LayerNorm2d`

作用：

- 对应原仓库的自定义二维 LayerNorm

为什么不是直接调用现成 `LayerNorm`：

- 原仓库并不是标准 `nn.LayerNorm(N, C, H, W)` 用法
- 它是“按通道维做归一化”的自定义实现
- 为了等价迁移，必须保留原始公式

## 3.5 `jittor_impl/models/layers_jt.py`

完整代码：

```python
"""Skeletons for the next Jittor migration step.

Step 1 only creates the dependency split points so the upcoming model
migration can land in stable files without touching the PyTorch tree.
The actual layer implementations are planned for the next step.
"""

from __future__ import annotations

import jittor as jt
from jittor import nn


class SimpleASPP(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def execute(self, x: jt.Var) -> jt.Var:
        raise NotImplementedError("SimpleASPP will be migrated in the next step.")


class DifferenceAwareOps(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def execute(self, x: jt.Var) -> jt.Var:
        raise NotImplementedError("DifferenceAwareOps will be migrated in the next step.")


class RGPU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def execute(self, x: jt.Var) -> jt.Var:
        raise NotImplementedError("RGPU will be migrated in the next step.")


class MHSIU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def execute(self, l: jt.Var, m: jt.Var, s: jt.Var) -> jt.Var:
        del l, m, s
        raise NotImplementedError("MHSIU will be migrated in the next step.")
```

解释：

这个文件现在还不是等价迁移，只是把后续 Jittor 层级的文件落点固定下来，便于下一轮直接把原仓库的 `methods/zoomnext/layers.py` 逐类搬过来。

当前状态说明：

- 这是占位代码
- 还不能用于训练或推理
- 后续必须替换成完整等价实现

## 3.6 `jittor_impl/models/backbone/resnet50_jt.py`

完整代码：

```python
"""ResNet50 backbone placeholder for the upcoming Jittor migration step."""

from __future__ import annotations


def build_resnet50(pretrained: bool = False, weight_path: str | None = None):
    del pretrained, weight_path
    raise NotImplementedError("ResNet50 migration is planned for the next step.")


def extract_features(x):
    del x
    raise NotImplementedError("ResNet50 migration is planned for the next step.")
```

解释：

这个文件当前只是为 `resnet50` backbone 的后续迁移占位。  
真正完整迁移时，它必须至少提供：

- backbone 构建
- 本地 `resnet50-timm.pth` 权重加载
- 输出 `c1-c5` 五级特征

当前状态仍然不是等价实现。

## 3.7 `jittor_impl/models/zoomnext_jt.py`

完整代码：

```python
"""ZoomNeXt model placeholder for the upcoming Jittor migration step."""

from __future__ import annotations


class _ZoomNeXt_Base:
    pass


class RN50_ZoomNeXt_JT(_ZoomNeXt_Base):
    def __init__(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError("RN50_ZoomNeXt_JT will be migrated in the next step.")
```

解释：

这个文件当前也是占位。它的作用是先确定：

- Jittor 版整网文件位置
- `RN50_ZoomNeXt_JT` 类名
- 后续 `_ZoomNeXt_Base` 的承载位置

但当前它还没有任何实际前向逻辑，因此还不能视为模型迁移完成。

## 3.8 `scripts/validate_jittor_ops.py`

完整代码：

```python
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
```

解释：

这个脚本是这轮实验最关键的验证工具。

它做了 4 件事：

1. 自动把仓库根目录加入 `sys.path`
2. 分别导入 PyTorch 版和 Jittor 版基础算子
3. 用同一组输入和同一组 `LayerNorm2d` 参数做数值比较
4. 输出每个算子的最大绝对误差和平均绝对误差，并按阈值判断通过或失败

为什么要加 `sys.path` 修复：

你在 Ubuntu 容器里执行的是：

```bash
python3 scripts/validate_jittor_ops.py
```

这种执行方式下，Python 默认只把 `scripts/` 目录加到导入路径里，导致原来的：

```python
from methods.zoomnext.ops import ...
```

报出：

```text
ModuleNotFoundError: No module named 'methods'
```

所以这里加了：

```python
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
```

这个修复的作用是：

- 无论脚本从哪里执行
- 只要脚本还在仓库 `scripts/` 下
- 都能找到仓库根目录里的 `methods/` 和 `jittor_impl/`

## 4. 本轮验证结果

你在 Ubuntu 容器中执行了：

```bash
python3 scripts/validate_jittor_ops.py
```

得到的核心结果如下：

```json
[
  {
    "name": "resize_to",
    "shape": [2, 3, 11, 13],
    "max_abs_err": 2.384185791015625e-07,
    "mean_abs_err": 2.020809830582948e-08
  },
  {
    "name": "rescale_2x",
    "shape": [2, 3, 30, 38],
    "max_abs_err": 2.384185791015625e-07,
    "mean_abs_err": 1.8990538919183564e-08
  },
  {
    "name": "PixelNormalizer",
    "shape": [2, 3, 15, 19],
    "max_abs_err": 9.5367431640625e-07,
    "mean_abs_err": 9.81725634119357e-08
  },
  {
    "name": "LayerNorm2d",
    "shape": [2, 3, 15, 19],
    "max_abs_err": 3.635883331298828e-06,
    "mean_abs_err": 8.053584821254844e-08
  }
]
```

以及最终结论：

```text
Validation passed.
```

## 5. 这说明什么

这个验证结果说明：

### 5.1 已经通过验证的内容

Jittor 版以下基础算子和 PyTorch 版数值非常接近：

- `resize_to`
- `rescale_2x`
- `PixelNormalizer`
- `LayerNorm2d`

从误差量级看：

- 最大误差都在 `1e-6` 到 `1e-7` 量级
- 平均误差都在 `1e-8` 量级

这基本可以认为：

- 插值行为对齐了
- 输入归一化行为对齐了
- 自定义 `LayerNorm2d` 行为也对齐了

### 5.2 还不能说明的内容

这个结果还不能说明下面这些已经对齐：

- `SimpleASPP`
- `MHSIU`
- `DifferenceAwareOps`
- `RGPU`
- `ResNet50` backbone 特征提取
- `RN50_ZoomNeXt` 整体前向
- loss、训练、权重加载

所以当前结论必须准确表达为：

> “Jittor 基础算子层的第一轮迁移是成功的，并通过了 Ubuntu 容器中的数值验证；但整个 `RN50_ZoomNeXt` 还没有完成等价迁移。”

## 6. 当前存在的问题与后续要求

你新增的要求是：

> 每次完成迁移新代码、修复迁移代码中的错误、或收到新的验证结果之后，都要把最新内容追加到这份实验报告里。

这个要求从现在开始生效。  
后续每一轮我都会在这份文件里补充：

- 本轮新增/修改的代码
- 代码块形式的完整代码
- 每段代码的解释
- 新发现的问题
- 修复内容
- 最新验证命令
- 最新验证输出
- 当前是否达到“等价迁移”的结论

## 7. 当前阶段结论

当前阶段可以认为：

- 迁移目录骨架已建立
- 基础算子层已开始迁移
- 基础算子层验证通过
- 但完整等价迁移尚未完成

离“与原仓库等价迁移”还差以下关键步骤：

1. 完整迁移 `methods/zoomnext/layers.py`
2. 完整迁移 `ResNet50` backbone
3. 完整迁移 `RN50_ZoomNeXt` 前向
4. 增加 backbone 权重加载与逐层对比
5. 增加整网 forward 对比
6. 最后再进入 loss 与训练链路
