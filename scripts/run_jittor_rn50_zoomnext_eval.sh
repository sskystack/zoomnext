#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="configs/icod_train.py"
DATA_CFG="./dataset.yaml"
OUTPUT_DIR="outputs_jittor_eval"
BATCH_SIZE="4"
ENCODER_WEIGHT_PATH="pretrained_weights/resnet50-timm.pth"
USE_CUDA=1
SAVE_RESULTS=1
PRETRAINED=0
CKPT=""
TEST_DATASETS=("chameleon" "camo_te" "cod10k_te" "nc4k")

usage() {
  cat <<'EOF'
用法:
  bash scripts/run_jittor_rn50_zoomnext_eval.sh --ckpt <checkpoint.pkl> [可选参数]

可选参数:
  --config <path>                配置文件路径，默认 configs/icod_train.py
  --data-cfg <path>              数据集配置路径，默认 ./dataset.yaml
  --output-dir <path>            输出目录，默认 outputs_jittor_eval
  --batch-size <int>             测试 batch size，默认 4
  --encoder-weight-path <path>   ResNet50 预训练权重路径
  --test-datasets <names...>     测试集列表，默认 chameleon camo_te cod10k_te nc4k
  --cpu                          使用 CPU 推理
  --no-save-results              不保存预测图，只计算指标
  --pretrained                   初始化模型时加载 encoder 预训练权重
  -h, --help                     显示帮助

示例:
  bash scripts/run_jittor_rn50_zoomnext_eval.sh \
    --ckpt outputs_jittor/.../step_000002.pkl \
    --batch-size 4 \
    --test-datasets chameleon camo_te cod10k_te nc4k
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt)
      CKPT="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --data-cfg)
      DATA_CFG="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --encoder-weight-path)
      ENCODER_WEIGHT_PATH="$2"
      shift 2
      ;;
    --test-datasets)
      shift
      TEST_DATASETS=()
      while [[ $# -gt 0 && "${1}" != --* ]]; do
        TEST_DATASETS+=("$1")
        shift
      done
      ;;
    --cpu)
      USE_CUDA=0
      shift
      ;;
    --no-save-results)
      SAVE_RESULTS=0
      shift
      ;;
    --pretrained)
      PRETRAINED=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "未知参数: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${CKPT}" ]]; then
  echo "缺少 --ckpt 参数" >&2
  usage
  exit 1
fi

CMD=(
  python3 scripts/test_jittor_rn50_zoomnext.py
  --config "${CONFIG}"
  --data-cfg "${DATA_CFG}"
  --load-from "${CKPT}"
  --output-dir "${OUTPUT_DIR}"
  --batch-size "${BATCH_SIZE}"
  --encoder-weight-path "${ENCODER_WEIGHT_PATH}"
  --test-datasets "${TEST_DATASETS[@]}"
)

if [[ "${SAVE_RESULTS}" == "1" ]]; then
  CMD+=(--save-results)
fi

if [[ "${PRETRAINED}" == "1" ]]; then
  CMD+=(--pretrained)
fi

if [[ "${USE_CUDA}" == "1" ]]; then
  CMD+=(--use-cuda)
fi

printf '执行命令:\n%s\n' "${CMD[*]}"
"${CMD[@]}"
