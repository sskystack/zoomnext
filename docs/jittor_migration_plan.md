# ZoomNeXt Jittor Migration Plan

## Goal

Migrate the full ZoomNeXt project from the current PyTorch implementation to Jittor while preserving:

- image training
- image evaluation
- video fine-tuning
- pretrained backbone loading
- optimizer and scheduler behavior
- metric evaluation workflow

The migration should be done in parallel with the current PyTorch codebase instead of rewriting the existing files in place.

## Recommended Strategy

Build a new `jittor_impl/` tree and keep the current PyTorch version as the reference implementation.

Reasons:

- the current repository already has stable training and evaluation entrypoints
- PyTorch outputs can be used as the correctness baseline
- backbone migration and weight conversion are high risk, so side-by-side comparison is safer
- unrelated existing code and scripts can remain untouched during migration

Recommended top-level layout:

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

## Migration Order

### Phase 1: Image inference baseline

Target:

- build Jittor common ops
- build Jittor ZoomNeXt layers
- build Jittor ResNet50 backbone
- run `RN50_ZoomNeXt` image forward successfully

This is the lowest-risk entry point because `RN50_ZoomNeXt` is the simplest backbone path in the repo.

### Phase 2: Image training baseline

Target:

- migrate image dataset pipeline
- migrate loss, optimizer, scheduler, checkpoint save/load
- run one training step and one short training job in Jittor

### Phase 3: Backbone expansion

Target:

- add PVTv2 B2/B3/B4/B5
- add EfficientNet B1/B4
- add pretrained weight conversion and loading

Suggested order:

1. `ResNet50`
2. `PVTv2`
3. `EfficientNet`

### Phase 4: Video pipeline

Target:

- migrate video datasets
- migrate `videoPvtV2B5_ZoomNeXt`
- reproduce video fine-tuning and evaluation

This phase should be last because it depends on all earlier model and training pieces being stable.

## Files To Create First

These files should be created first, in this order.

### 1. `jittor_impl/models/ops_jt.py`

Write first:

- `resize_to(x, tgt_hw)`
- `rescale_2x(x, scale_factor=2)`
- `global_avgpool(x)`
- `ConvBNReLU`
- `PixelNormalizer`
- `LayerNorm2d`

Final interface list:

- `resize_to`
- `rescale_2x`
- `global_avgpool`
- `ConvBN`
- `CBR`
- `ConvBNReLU`
- `ConvGNReLU`
- `PixelNormalizer`
- `LayerNorm2d`

PyTorch correspondence:

- `methods/zoomnext/ops.py`
- `utils/ops/tensor_ops.py`

Why first:

- every layer and model depends on these operators

### 2. `jittor_impl/models/layers_jt.py`

Write first:

- `SimpleASPP`
- `MHSIU`
- `DifferenceAwareOps`
- `RGPU`

Final interface list:

- `SimpleASPP`
- `DifferenceAwareOps`
- `RGPU`
- `MHSIU`

PyTorch correspondence:

- `methods/zoomnext/layers.py`

Why second:

- these are the core ZoomNeXt building blocks shared by all backbones

### 3. `jittor_impl/models/backbone/resnet50_jt.py`

Write first:

- `build_resnet50(pretrained=False, weight_path=None)`
- `extract_features(x)`

Final interface list:

- `build_resnet50`
- `load_resnet50_pretrained`
- `extract_features`

PyTorch correspondence:

- `methods/zoomnext/zoomnext.py` (`RN50_ZoomNeXt` encoder path)

Why third:

- `RN50_ZoomNeXt` is the easiest end-to-end model to validate

### 4. `jittor_impl/models/zoomnext_jt.py`

Write first:

- `_ZoomNeXt_Base`
- `RN50_ZoomNeXt_JT`

Final interface list:

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

PyTorch correspondence:

- `methods/zoomnext/zoomnext.py`

Why fourth:

- this gives the first usable Jittor model for forward and loss validation

### 5. `jittor_impl/data/transforms.py`

Write first:

- `build_image_train_transform()`
- `build_video_frame_transform()`
- `build_video_shared_transform()`

Final interface list:

- `build_image_train_transform`
- `build_video_frame_transform`
- `build_video_shared_transform`
- `replay_shared_transform`

PyTorch correspondence:

- image augmentation code in `main_for_image.py`
- video augmentation code in `main_for_video.py`

### 6. `jittor_impl/data/image_dataset.py`

Write first:

- `build_ms_inputs(image, base_h, base_w)`
- `ImageTrainDatasetJT`
- `ImageTestDatasetJT`

Final interface list:

- `build_ms_inputs`
- `ImageTrainDatasetJT`
- `ImageTestDatasetJT`

PyTorch correspondence:

- `main_for_image.py`
- `utils/io/image.py`
- `utils/ops/array_ops.py`

### 7. `jittor_impl/engine/losses.py`

Write first:

- `get_ual_coef(iter_percentage, method="cos", milestones=(0, 1))`
- `compute_sod_loss(logits, mask, iter_percentage)`

Final interface list:

- `compute_bce_loss`
- `compute_ual_loss`
- `compute_sod_loss`
- `get_ual_coef`

PyTorch correspondence:

- `_ZoomNeXt_Base.forward` in `methods/zoomnext/zoomnext.py`

### 8. `jittor_impl/engine/optimizer.py`

Write first:

- `group_params(model, group_mode, initial_lr, optim_cfg)`
- `construct_optimizer(model, initial_lr, mode, group_mode, cfg)`

Final interface list:

- `get_optimizer`
- `group_params`
- `construct_optimizer`
- `get_lr_groups`
- `get_lr_strings`

PyTorch correspondence:

- `utils/pipeline/optimizer.py`

### 9. `jittor_impl/engine/scheduler.py`

Write first:

- `Scheduler`
- `get_scheduler_coef_func`
- `get_warmup_coef_func`

Final interface list:

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

PyTorch correspondence:

- `utils/pipeline/scheduler.py`

### 10. `jittor_impl/main_for_image_jt.py`

Write first:

- `parse_cfg()`
- `train(model, cfg)`
- `test(model, cfg)`
- `main()`

Final interface list:

- `ImageTrainDatasetJT` integration
- `ImageTestDatasetJT` integration
- evaluator integration
- training loop
- evaluation loop
- checkpoint save/load entrypoints

PyTorch correspondence:

- `main_for_image.py`

This is the first complete executable Jittor entrypoint that should run.

## Full File Checklist

### Models

#### `jittor_impl/models/ops_jt.py`

- Maps to:
  - `methods/zoomnext/ops.py`
  - `utils/ops/tensor_ops.py`
- Interfaces to implement:
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

- Maps to:
  - `methods/zoomnext/layers.py`
- Interfaces to implement:
  - `SimpleASPP`
  - `DifferenceAwareOps`
  - `RGPU`
  - `MHSIU`

#### `jittor_impl/models/backbone/resnet50_jt.py`

- Maps to:
  - `methods/zoomnext/zoomnext.py` ResNet path
- Interfaces to implement:
  - `build_resnet50`
  - `load_resnet50_pretrained`
  - `extract_features`

#### `jittor_impl/models/backbone/pvtv2_jt.py`

- Maps to:
  - `methods/backbone/pvt_v2_eff.py`
  - `methods/zoomnext/zoomnext.py` PVTv2 model classes
- Interfaces to implement:
  - `pvt_v2_b2`
  - `pvt_v2_b3`
  - `pvt_v2_b4`
  - `pvt_v2_b5`
  - `load_pvt_pretrained`

#### `jittor_impl/models/backbone/efficientnet_jt.py`

- Maps to:
  - `methods/backbone/efficientnet.py`
  - `methods/backbone/efficientnet_utils.py`
  - `methods/zoomnext/zoomnext.py` EfficientNet model classes
- Interfaces to implement:
  - `efficientnet_b1`
  - `efficientnet_b4`
  - `load_efficientnet_pretrained`
  - `extract_endpoints`

#### `jittor_impl/models/zoomnext_jt.py`

- Maps to:
  - `methods/zoomnext/zoomnext.py`
- Interfaces to implement:
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

### Data

#### `jittor_impl/data/array_ops_jt.py`

- Maps to:
  - `utils/ops/array_ops.py`
- Interfaces to implement:
  - `minmax`
  - `clip_to_normalize`
  - `save_array_as_image`
  - `resize`
  - `ms_resize`

#### `jittor_impl/data/io_jt.py`

- Maps to:
  - `utils/io/image.py`
  - `utils/io/__init__.py`
- Interfaces to implement:
  - `read_gray_array`
  - `read_color_array`
  - `save_array_as_image`

#### `jittor_impl/data/transforms.py`

- Maps to:
  - transform code in `main_for_image.py`
  - transform code in `main_for_video.py`
- Interfaces to implement:
  - `build_image_train_transform`
  - `build_video_frame_transform`
  - `build_video_shared_transform`
  - `replay_shared_transform`

#### `jittor_impl/data/image_dataset.py`

- Maps to:
  - `main_for_image.py`
- Interfaces to implement:
  - `build_ms_inputs`
  - `ImageTrainDatasetJT`
  - `ImageTestDatasetJT`

#### `jittor_impl/data/video_dataset.py`

- Maps to:
  - `main_for_video.py`
- Interfaces to implement:
  - `get_number_from_tail`
  - `VideoTrainDatasetJT`
  - `VideoTestDatasetJT`

### Engine

#### `jittor_impl/engine/losses.py`

- Maps to:
  - loss path in `methods/zoomnext/zoomnext.py`
- Interfaces to implement:
  - `compute_bce_loss`
  - `compute_ual_loss`
  - `compute_sod_loss`
  - `get_ual_coef`

#### `jittor_impl/engine/optimizer.py`

- Maps to:
  - `utils/pipeline/optimizer.py`
- Interfaces to implement:
  - `get_optimizer`
  - `group_params`
  - `construct_optimizer`
  - `get_lr_groups`
  - `get_lr_strings`

#### `jittor_impl/engine/scheduler.py`

- Maps to:
  - `utils/pipeline/scheduler.py`
- Interfaces to implement:
  - `Scheduler`
  - all scheduler coefficient helper functions

#### `jittor_impl/engine/checkpoint.py`

- Maps to:
  - `utils/io/params.py`
- Interfaces to implement:
  - `save_weight_jt`
  - `load_weight_jt`
  - `load_partial_weight_jt`

#### `jittor_impl/engine/evaluator.py`

- Maps to:
  - evaluator logic in `main_for_image.py`
  - evaluator logic in `main_for_video.py`
  - metric calling in `utils/recorder/group_metric_caller.py`
- Interfaces to implement:
  - `ImageEvaluatorJT`
  - `VideoEvaluatorJT`

#### `jittor_impl/engine/utils.py`

- Maps to:
  - `utils/pt_utils.py`
  - `utils/py_utils.py`
- Interfaces to implement:
  - `set_seed`
  - `to_jt_var`
  - `freeze_bn_stats`
  - `construct_path_like_pt`

### Entrypoints

#### `jittor_impl/main_for_image_jt.py`

- Maps to:
  - `main_for_image.py`
- Interfaces to implement:
  - `parse_cfg`
  - `train`
  - `test`
  - `main`

#### `jittor_impl/main_for_video_jt.py`

- Maps to:
  - `main_for_video.py`
- Interfaces to implement:
  - `parse_cfg`
  - `train`
  - `test`
  - `main`

### Tools

#### `jittor_impl/tools/convert_pt_to_jt.py`

- New tooling required for migration
- Interfaces to implement:
  - `load_pt_state_dict`
  - `map_pt_key_to_jt_key`
  - `convert_tensor`
  - `convert_checkpoint`
  - `save_jt_checkpoint`

#### `jittor_impl/tools/compare_pt_jt.py`

- New tooling required for validation
- Interfaces to implement:
  - `compare_ops`
  - `compare_layers`
  - `compare_model_forward`
  - `compare_train_step`
  - `compare_eval_subset`

## Correspondence Matrix

| Jittor file | PyTorch reference |
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

## Validation Plan Against PyTorch

Validation should be done in layers. Do not start with full training first.

### 1. Operator-level validation

Compare PyTorch and Jittor outputs for:

- `resize_to`
- `PixelNormalizer`
- `ConvBNReLU`
- `LayerNorm2d`

Method:

- fix random seed
- generate the same tensor input
- copy matching weights
- compare outputs

Pass criteria:

- `max_abs_err < 1e-5`
- `mean_abs_err < 1e-6`

### 2. Layer-level validation

Compare:

- `SimpleASPP`
- `MHSIU`
- `DifferenceAwareOps`
- `RGPU`

Method:

- export PyTorch submodule weights
- load them into the Jittor counterpart
- run the same input through both

Pass criteria:

- `max_abs_err < 1e-4`

Notes:

- `DifferenceAwareOps` is the most sensitive because it uses reshape/rearrange, temporal rolling, and einsum

### 3. Backbone feature validation

Compare feature maps stage by stage.

For `ResNet50`, compare the 5 extracted feature stages.

For `PVTv2` and `EfficientNet`, compare the reduction endpoints.

Pass criteria:

- cosine similarity of each stage output `> 0.999`
- relative error `< 1e-4`

### 4. Whole-model forward validation

Compare the final `logits` of:

- `RN50_ZoomNeXt`
- later `PvtV2B*_ZoomNeXt`
- later `EffB*_ZoomNeXt`

Method:

- construct identical `image_s`, `image_m`, `image_l`
- run the model in eval mode on both frameworks

Pass criteria:

- `max_abs_err < 1e-4`

If this fails, dump and compare intermediate tensors in order:

1. normalized inputs
2. encoder features
3. `tra_5`
4. `siu_5`
5. `hmu_5`
6. repeated decoder stages
7. predictor output

### 5. Loss validation

Compare:

- BCE loss
- UAL loss
- total loss

Method:

- use the same logits and masks
- run loss calculation in both frameworks

Pass criteria:

- each loss term relative error `< 1e-3`

### 6. Single-step training validation

Method:

- same batch
- same initial weights
- same optimizer settings
- one update step in PyTorch and Jittor

Compare:

- parameter deltas
- loss before update
- loss after update

Pass criteria:

- update direction cosine similarity `> 0.999`
- loss change trend should match

This step is important for catching mismatches in:

- parameter grouping
- weight decay handling
- scheduler stepping
- BN freezing behavior

### 7. Small evaluation-set validation

Method:

- select a small subset, such as 20 images
- run inference in both frameworks
- save prediction maps
- compute the same metrics

Compare:

- per-pixel prediction differences
- `sm`
- `wfm`
- `mae`
- `em`
- `fmeasure`

Pass criteria:

- metric absolute difference ideally within `0.001` to `0.01`

### 8. Short training-run validation

Method:

- run 1 to 3 epochs with the same config
- compare:
  - learning rate curve
  - per-iteration loss
  - average loss
  - validation metric trend

Pass criteria:

- trend should be consistent
- validation metrics should stay close to the PyTorch baseline

## Practical Validation Rules

- validate in `eval()` mode before validating in `train()` mode
- validate FP32 first, AMP later
- add optional intermediate tensor dumping to every major Jittor model block
- implement partial checkpoint conversion to isolate failures by submodule
- do not start video migration until image migration is stable

## First-Week Execution Plan

### Day 1

- create `jittor_impl/models/ops_jt.py`
- create `jittor_impl/models/layers_jt.py`
- validate operator-level and layer-level outputs

### Day 2

- create `jittor_impl/models/backbone/resnet50_jt.py`
- create `jittor_impl/models/zoomnext_jt.py` with `RN50_ZoomNeXt_JT`
- validate full forward path for image inference

### Day 3

- create `jittor_impl/data/array_ops_jt.py`
- create `jittor_impl/data/io_jt.py`
- create `jittor_impl/data/image_dataset.py`
- create `jittor_impl/engine/losses.py`

### Day 4

- create `jittor_impl/engine/optimizer.py`
- create `jittor_impl/engine/scheduler.py`
- create `jittor_impl/engine/checkpoint.py`
- create `jittor_impl/main_for_image_jt.py`

### Day 5

- create `jittor_impl/tools/convert_pt_to_jt.py`
- create `jittor_impl/tools/compare_pt_jt.py`
- run single-step training comparison and small-set evaluation comparison

### After Week 1

- add `pvtv2_jt.py`
- add `efficientnet_jt.py`
- add `main_for_video_jt.py`
- add `video_dataset.py`
- migrate video fine-tuning

## Definition of Done

The Jittor migration can be considered correct only when all of the following are true:

- all target Jittor entrypoints can run
- all required backbones can load pretrained weights
- image training reproduces the PyTorch training behavior closely
- image evaluation metrics are close to the PyTorch baseline
- video fine-tuning reproduces the PyTorch behavior closely
- comparison tooling can identify layer and checkpoint mismatches quickly
