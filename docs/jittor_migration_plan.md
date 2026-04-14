# ZoomNeXt 迁移到 Jittor 的实施清单

## 目标

把当前 ZoomNeXt 项目从 PyTorch 完整迁移到 Jittor，并尽量保持以下能力不变：

- 图像训练
- 图像测试
- 视频微调
- 预训练骨干网络加载
- 优化器与学习率调度行为
- 指标评估流程

迁移方式不建议直接改写现有 PyTorch 文件，而是并行新增一套 `jittor_impl/` 实现，把当前仓库作为对照基线。

## 总体策略

建议新建如下目录：

```text
jittor_impl/
  data/
  engine/
  models/
    backbone/
  tools/
  main_for_image_jt.py
  main_for_video_jt.py
```

这样做的原因：

- 现有 PyTorch 代码已经是稳定参考实现
- Jittor 版本可以逐层和 PyTorch 做 A/B 对照
- backbone 和预训练权重迁移风险最高，平行实现更容易定位问题
- 不会干扰当前训练脚本、配置文件和实验流程

## 推荐迁移顺序

### 第一阶段：先打通图像推理

目标：

- 完成 Jittor 版基础算子
- 完成 Jittor 版 ZoomNeXt 核心层
- 完成 Jittor 版 ResNet50 backbone
- 先让 `RN50_ZoomNeXt` 在图像推理上跑通

这是最稳的入口，因为 `RN50_ZoomNeXt` 的骨干路径最简单。

### 第二阶段：再打通图像训练

目标：

- 迁移图像数据集
- 迁移 loss、optimizer、scheduler、checkpoint
- 跑通单步训练和短程训练

### 第三阶段：扩展骨干网络

目标：

- 加入 PVTv2 B2/B3/B4/B5
- 加入 EfficientNet B1/B4
- 加入预训练权重转换和加载

推荐顺序：

1. `ResNet50`
2. `PVTv2`
3. `EfficientNet`

### 第四阶段：迁移视频链路

目标：

- 迁移视频数据集
- 迁移 `videoPvtV2B5_ZoomNeXt`
- 跑通视频微调与评估

视频部分依赖前面模型、训练器、权重加载都稳定，所以应放到最后。

## 第一批先新建的文件

下面这些文件建议优先创建，顺序也尽量按这个来。

### 1. `jittor_impl/models/ops_jt.py`

先写这些接口：

- `resize_to(x, tgt_hw)`
- `rescale_2x(x, scale_factor=2)`
- `global_avgpool(x)`
- `ConvBNReLU`
- `PixelNormalizer`
- `LayerNorm2d`

最终要有的接口：

- `resize_to`
- `rescale_2x`
- `global_avgpool`
- `ConvBN`
- `CBR`
- `ConvBNReLU`
- `ConvGNReLU`
- `PixelNormalizer`
- `LayerNorm2d`

对应的 PyTorch 文件：

- `methods/zoomnext/ops.py`
- `utils/ops/tensor_ops.py`

优先原因：

- 所有 layer 和 model 都依赖这层

### 2. `jittor_impl/models/layers_jt.py`

先写这些接口：

- `SimpleASPP`
- `MHSIU`
- `DifferenceAwareOps`
- `RGPU`

最终要有的接口：

- `SimpleASPP`
- `DifferenceAwareOps`
- `RGPU`
- `MHSIU`

对应的 PyTorch 文件：

- `methods/zoomnext/layers.py`

优先原因：

- 这是 ZoomNeXt 主体结构的核心模块

### 3. `jittor_impl/models/backbone/resnet50_jt.py`

先写这些接口：

- `build_resnet50(pretrained=False, weight_path=None)`
- `extract_features(x)`

最终要有的接口：

- `build_resnet50`
- `load_resnet50_pretrained`
- `extract_features`

对应的 PyTorch 位置：

- `methods/zoomnext/zoomnext.py` 里 `RN50_ZoomNeXt` 的 encoder 部分

优先原因：

- `RN50_ZoomNeXt` 最适合做第一套端到端验证

### 4. `jittor_impl/models/zoomnext_jt.py`

先写这些接口：

- `_ZoomNeXt_Base`
- `RN50_ZoomNeXt_JT`

最终要有的接口：

- `_ZoomNeXt_Base`
- `RN50_ZoomNeXt_JT`
- `PvtV2B2_ZoomNeXt_JT`
- `PvtV2B3_ZoomNeXt_JT`
- `PvtV2B4_ZoomNeXt_JT`
- `PvtV2B5_ZoomNeXt_JT`
- `EffB1_ZoomNeXt_JT`
- `EffB4_ZoomNeXt_JT`
- `videoPvtV2B5_ZoomNeXt_JT`
- `get_grouped_params`

对应的 PyTorch 文件：

- `methods/zoomnext/zoomnext.py`

优先原因：

- 这是第一版可运行 Jittor 模型的主体文件

### 5. `jittor_impl/data/transforms.py`

先写这些接口：

- `build_image_train_transform()`
- `build_video_frame_transform()`
- `build_video_shared_transform()`

最终要有的接口：

- `build_image_train_transform`
- `build_video_frame_transform`
- `build_video_shared_transform`
- `replay_shared_transform`

对应的 PyTorch 位置：

- `main_for_image.py` 中图像增强逻辑
- `main_for_video.py` 中视频增强逻辑

### 6. `jittor_impl/data/image_dataset.py`

先写这些接口：

- `build_ms_inputs(image, base_h, base_w)`
- `ImageTrainDatasetJT`
- `ImageTestDatasetJT`

最终要有的接口：

- `build_ms_inputs`
- `ImageTrainDatasetJT`
- `ImageTestDatasetJT`

对应的 PyTorch 文件：

- `main_for_image.py`
- `utils/io/image.py`
- `utils/ops/array_ops.py`

### 7. `jittor_impl/engine/losses.py`

先写这些接口：

- `get_ual_coef(iter_percentage, method="cos", milestones=(0, 1))`
- `compute_sod_loss(logits, mask, iter_percentage)`

最终要有的接口：

- `compute_bce_loss`
- `compute_ual_loss`
- `compute_sod_loss`
- `get_ual_coef`

对应的 PyTorch 位置：

- `methods/zoomnext/zoomnext.py` 中 `_ZoomNeXt_Base.forward`

### 8. `jittor_impl/engine/optimizer.py`

先写这些接口：

- `group_params(model, group_mode, initial_lr, optim_cfg)`
- `construct_optimizer(model, initial_lr, mode, group_mode, cfg)`

最终要有的接口：

- `get_optimizer`
- `group_params`
- `construct_optimizer`
- `get_lr_groups`
- `get_lr_strings`

对应的 PyTorch 文件：

- `utils/pipeline/optimizer.py`

### 9. `jittor_impl/engine/scheduler.py`

先写这些接口：

- `Scheduler`
- `get_scheduler_coef_func`
- `get_warmup_coef_func`

最终要有的接口：

- `Scheduler`
- `linear_increase`
- `cos_anneal`
- `poly_anneal`
- `linear_anneal`
- `get_f3_coef_func`
- `get_step_coef_func`
- `get_cos_coef_func`
- `get_fatcos_coef_func`
- `get_poly_coef_func`
- `get_scheduler_coef_func`
- `get_warmup_coef_func`

对应的 PyTorch 文件：

- `utils/pipeline/scheduler.py`

### 10. `jittor_impl/main_for_image_jt.py`

先写这些接口：

- `parse_cfg()`
- `train(model, cfg)`
- `test(model, cfg)`
- `main()`

最终要有的能力：

- 接入 `ImageTrainDatasetJT`
- 接入 `ImageTestDatasetJT`
- 接入 evaluator
- 接入 checkpoint 保存与加载
- 能跑图像训练和图像测试

对应的 PyTorch 文件：

- `main_for_image.py`

这会是第一套完整可执行的 Jittor 入口。

## 全量文件清单

### 一、模型部分

#### `jittor_impl/models/ops_jt.py`

对应 PyTorch：

- `methods/zoomnext/ops.py`
- `utils/ops/tensor_ops.py`

需要实现的接口：

- `resize_to`
- `rescale_2x`
- `global_avgpool`
- `ConvBN`
- `CBR`
- `ConvBNReLU`
- `ConvGNReLU`
- `PixelNormalizer`
- `LayerNorm2d`

#### `jittor_impl/models/layers_jt.py`

对应 PyTorch：

- `methods/zoomnext/layers.py`

需要实现的接口：

- `SimpleASPP`
- `DifferenceAwareOps`
- `RGPU`
- `MHSIU`

#### `jittor_impl/models/backbone/resnet50_jt.py`

对应 PyTorch：

- `methods/zoomnext/zoomnext.py` 中 ResNet 路径

需要实现的接口：

- `build_resnet50`
- `load_resnet50_pretrained`
- `extract_features`

#### `jittor_impl/models/backbone/pvtv2_jt.py`

对应 PyTorch：

- `methods/backbone/pvt_v2_eff.py`
- `methods/zoomnext/zoomnext.py` 中 PVTv2 相关类

需要实现的接口：

- `pvt_v2_b2`
- `pvt_v2_b3`
- `pvt_v2_b4`
- `pvt_v2_b5`
- `load_pvt_pretrained`

#### `jittor_impl/models/backbone/efficientnet_jt.py`

对应 PyTorch：

- `methods/backbone/efficientnet.py`
- `methods/backbone/efficientnet_utils.py`
- `methods/zoomnext/zoomnext.py` 中 EfficientNet 相关类

需要实现的接口：

- `efficientnet_b1`
- `efficientnet_b4`
- `load_efficientnet_pretrained`
- `extract_endpoints`

#### `jittor_impl/models/zoomnext_jt.py`

对应 PyTorch：

- `methods/zoomnext/zoomnext.py`

需要实现的接口：

- `_ZoomNeXt_Base`
- `RN50_ZoomNeXt_JT`
- `PvtV2B2_ZoomNeXt_JT`
- `PvtV2B3_ZoomNeXt_JT`
- `PvtV2B4_ZoomNeXt_JT`
- `PvtV2B5_ZoomNeXt_JT`
- `EffB1_ZoomNeXt_JT`
- `EffB4_ZoomNeXt_JT`
- `videoPvtV2B5_ZoomNeXt_JT`
- `get_grouped_params`

### 二、数据部分

#### `jittor_impl/data/array_ops_jt.py`

对应 PyTorch：

- `utils/ops/array_ops.py`

需要实现的接口：

- `minmax`
- `clip_to_normalize`
- `save_array_as_image`
- `resize`
- `ms_resize`

#### `jittor_impl/data/io_jt.py`

对应 PyTorch：

- `utils/io/image.py`
- `utils/io/__init__.py`

需要实现的接口：

- `read_gray_array`
- `read_color_array`
- `save_array_as_image`

#### `jittor_impl/data/transforms.py`

对应 PyTorch：

- `main_for_image.py`
- `main_for_video.py`

需要实现的接口：

- `build_image_train_transform`
- `build_video_frame_transform`
- `build_video_shared_transform`
- `replay_shared_transform`

#### `jittor_impl/data/image_dataset.py`

对应 PyTorch：

- `main_for_image.py`

需要实现的接口：

- `build_ms_inputs`
- `ImageTrainDatasetJT`
- `ImageTestDatasetJT`

#### `jittor_impl/data/video_dataset.py`

对应 PyTorch：

- `main_for_video.py`

需要实现的接口：

- `get_number_from_tail`
- `VideoTrainDatasetJT`
- `VideoTestDatasetJT`

### 三、训练与评估部分

#### `jittor_impl/engine/losses.py`

对应 PyTorch：

- `methods/zoomnext/zoomnext.py` 中 loss 路径

需要实现的接口：

- `compute_bce_loss`
- `compute_ual_loss`
- `compute_sod_loss`
- `get_ual_coef`

#### `jittor_impl/engine/optimizer.py`

对应 PyTorch：

- `utils/pipeline/optimizer.py`

需要实现的接口：

- `get_optimizer`
- `group_params`
- `construct_optimizer`
- `get_lr_groups`
- `get_lr_strings`

#### `jittor_impl/engine/scheduler.py`

对应 PyTorch：

- `utils/pipeline/scheduler.py`

需要实现的接口：

- `Scheduler`
- 各种 scheduler 系数函数

#### `jittor_impl/engine/checkpoint.py`

对应 PyTorch：

- `utils/io/params.py`

需要实现的接口：

- `save_weight_jt`
- `load_weight_jt`
- `load_partial_weight_jt`

#### `jittor_impl/engine/evaluator.py`

对应 PyTorch：

- `main_for_image.py` 中 evaluator
- `main_for_video.py` 中 evaluator
- `utils/recorder/group_metric_caller.py`

需要实现的接口：

- `ImageEvaluatorJT`
- `VideoEvaluatorJT`

#### `jittor_impl/engine/utils.py`

对应 PyTorch：

- `utils/pt_utils.py`
- `utils/py_utils.py`

需要实现的接口：

- `set_seed`
- `to_jt_var`
- `freeze_bn_stats`
- `construct_path_like_pt`

### 四、入口文件

#### `jittor_impl/main_for_image_jt.py`

对应 PyTorch：

- `main_for_image.py`

需要实现的接口：

- `parse_cfg`
- `train`
- `test`
- `main`

#### `jittor_impl/main_for_video_jt.py`

对应 PyTorch：

- `main_for_video.py`

需要实现的接口：

- `parse_cfg`
- `train`
- `test`
- `main`

### 五、迁移辅助工具

#### `jittor_impl/tools/convert_pt_to_jt.py`

这是迁移新增工具，没有现成一一对应的 PyTorch 文件。

需要实现的接口：

- `load_pt_state_dict`
- `map_pt_key_to_jt_key`
- `convert_tensor`
- `convert_checkpoint`
- `save_jt_checkpoint`

用途：

- 把 `.pth` 权重转换为 Jittor 可加载格式
- 支持整网转换和子模块转换

#### `jittor_impl/tools/compare_pt_jt.py`

这也是迁移新增工具。

需要实现的接口：

- `compare_ops`
- `compare_layers`
- `compare_model_forward`
- `compare_train_step`
- `compare_eval_subset`

用途：

- 用 PyTorch 结果作为基线，验证 Jittor 迁移正确性

## Jittor 文件与 PyTorch 文件对应关系

| Jittor 文件 | PyTorch 参考位置 |
| --- | --- |
| `jittor_impl/models/ops_jt.py` | `methods/zoomnext/ops.py`, `utils/ops/tensor_ops.py` |
| `jittor_impl/models/layers_jt.py` | `methods/zoomnext/layers.py` |
| `jittor_impl/models/backbone/resnet50_jt.py` | `methods/zoomnext/zoomnext.py` |
| `jittor_impl/models/backbone/pvtv2_jt.py` | `methods/backbone/pvt_v2_eff.py`, `methods/zoomnext/zoomnext.py` |
| `jittor_impl/models/backbone/efficientnet_jt.py` | `methods/backbone/efficientnet.py`, `methods/backbone/efficientnet_utils.py`, `methods/zoomnext/zoomnext.py` |
| `jittor_impl/models/zoomnext_jt.py` | `methods/zoomnext/zoomnext.py` |
| `jittor_impl/data/array_ops_jt.py` | `utils/ops/array_ops.py` |
| `jittor_impl/data/io_jt.py` | `utils/io/image.py` |
| `jittor_impl/data/transforms.py` | `main_for_image.py`, `main_for_video.py` |
| `jittor_impl/data/image_dataset.py` | `main_for_image.py` |
| `jittor_impl/data/video_dataset.py` | `main_for_video.py` |
| `jittor_impl/engine/losses.py` | `methods/zoomnext/zoomnext.py` |
| `jittor_impl/engine/optimizer.py` | `utils/pipeline/optimizer.py` |
| `jittor_impl/engine/scheduler.py` | `utils/pipeline/scheduler.py` |
| `jittor_impl/engine/checkpoint.py` | `utils/io/params.py` |
| `jittor_impl/engine/evaluator.py` | `main_for_image.py`, `main_for_video.py`, `utils/recorder/group_metric_caller.py` |
| `jittor_impl/engine/utils.py` | `utils/pt_utils.py`, `utils/py_utils.py` |
| `jittor_impl/main_for_image_jt.py` | `main_for_image.py` |
| `jittor_impl/main_for_video_jt.py` | `main_for_video.py` |

## 迁移后如何和 PyTorch 版本对比验证

不要一上来就跑完整训练，应该按层验证。

### 1. 算子级验证

先比较这些基础接口：

- `resize_to`
- `PixelNormalizer`
- `ConvBNReLU`
- `LayerNorm2d`

做法：

- 固定随机种子
- 构造同一份输入
- 复制相同权重
- 分别在 PyTorch 和 Jittor 下运行

通过标准：

- `max_abs_err < 1e-5`
- `mean_abs_err < 1e-6`

### 2. 模块级验证

逐个比较：

- `SimpleASPP`
- `MHSIU`
- `DifferenceAwareOps`
- `RGPU`

做法：

- 从 PyTorch 子模块导出权重
- 加载到对应 Jittor 子模块
- 同输入前向

通过标准：

- `max_abs_err < 1e-4`

重点说明：

- `DifferenceAwareOps` 最敏感，因为里面有 reshape、时间维移动、einsum

### 3. Backbone 特征级验证

逐 stage 对比特征图。

`ResNet50` 比较 5 个 stage 输出。  
`PVTv2` 和 `EfficientNet` 比较各个 reduction endpoint。

通过标准：

- 每层输出的 cosine similarity `> 0.999`
- relative error `< 1e-4`

### 4. 整网前向验证

比较以下模型最终 `logits`：

- `RN50_ZoomNeXt`
- 后续的 `PvtV2B*_ZoomNeXt`
- 后续的 `EffB*_ZoomNeXt`

做法：

- 构造完全相同的 `image_s`、`image_m`、`image_l`
- 两边都在 `eval` 模式下运行

通过标准：

- `max_abs_err < 1e-4`

如果不通过，按下面顺序逐层 dump 中间张量：

1. 归一化后的输入
2. encoder feature
3. `tra_5`
4. `siu_5`
5. `hmu_5`
6. 后续各 decoder stage
7. predictor 输出

### 5. Loss 对比验证

比较：

- BCE loss
- UAL loss
- total loss

做法：

- 使用同一组 logits 和 mask
- 分别在两边计算 loss

通过标准：

- 每项 loss 相对误差 `< 1e-3`

### 6. 单步训练验证

做法：

- 同一 batch
- 同一初始权重
- 同一 optimizer 配置
- 分别执行一步参数更新

比较内容：

- 参数更新量
- 更新前 loss
- 更新后 loss

通过标准：

- 参数更新方向 cosine similarity `> 0.999`
- loss 变化趋势一致

这一项特别容易发现以下问题：

- param group 分错
- weight decay 不一致
- scheduler 更新时机不一致
- BN freeze 行为不一致

### 7. 小样本评估验证

做法：

- 先选一小部分样本，比如 20 张图
- 两边分别推理
- 保存预测图
- 计算相同指标

比较内容：

- 逐像素预测差异
- `sm`
- `wfm`
- `mae`
- `em`
- `fmeasure`

通过标准：

- 指标绝对差值尽量控制在 `0.001` 到 `0.01`

### 8. 短程训练验证

做法：

- 用相同 config 各跑 1 到 3 个 epoch
- 比较：
  - 学习率曲线
  - 每 iter loss
  - 平均 loss
  - 验证集指标趋势

通过标准：

- 训练曲线趋势一致
- 验证指标接近 PyTorch 基线

## 验证时的实际规则

- 先验证 `eval()`，再验证 `train()`
- 先用 FP32，对齐后再考虑 AMP
- 每个关键 Jittor 模块最好支持导出中间张量
- checkpoint 转换脚本要支持“只转某个子模块”
- 图片链路稳定前，不要开始视频迁移

## 第一周执行计划

### 第 1 天

- 建 `jittor_impl/models/ops_jt.py`
- 建 `jittor_impl/models/layers_jt.py`
- 做算子级和模块级对比

### 第 2 天

- 建 `jittor_impl/models/backbone/resnet50_jt.py`
- 建 `jittor_impl/models/zoomnext_jt.py`，先只写 `RN50_ZoomNeXt_JT`
- 做图像前向对比

### 第 3 天

- 建 `jittor_impl/data/array_ops_jt.py`
- 建 `jittor_impl/data/io_jt.py`
- 建 `jittor_impl/data/image_dataset.py`
- 建 `jittor_impl/engine/losses.py`

### 第 4 天

- 建 `jittor_impl/engine/optimizer.py`
- 建 `jittor_impl/engine/scheduler.py`
- 建 `jittor_impl/engine/checkpoint.py`
- 建 `jittor_impl/main_for_image_jt.py`

### 第 5 天

- 建 `jittor_impl/tools/convert_pt_to_jt.py`
- 建 `jittor_impl/tools/compare_pt_jt.py`
- 做单步训练对比和小样本评估对比

### 第一周之后

- 补 `pvtv2_jt.py`
- 补 `efficientnet_jt.py`
- 补 `video_dataset.py`
- 补 `main_for_video_jt.py`
- 最后迁移视频微调

## 完成标准

只有在下面这些条件同时满足时，才算迁移基本正确：

- Jittor 版入口文件能够独立运行
- 目标 backbone 都能加载预训练权重
- 图像训练行为与 PyTorch 版本接近
- 图像评估指标与 PyTorch 基线接近
- 视频微调行为与 PyTorch 版本接近
- 对比脚本可以快速定位层级误差和权重映射问题
