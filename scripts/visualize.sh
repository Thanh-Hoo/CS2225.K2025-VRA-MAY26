#!/bin/bash
# Run every IDEA visualization: retrieval rank-list videos + CDA offset/attention figures.
#
# Usage:
#   ./scripts/visualize.sh                  # both datasets, both visualizations
#   ./scripts/visualize.sh RGBNT201         # one dataset only
#   ./scripts/visualize.sh MSVR310 maps     # one dataset, one visualization (video|maps|all)
#
# Override defaults via environment variables, e.g.:
#   NUM_QUERIES=-1 FPS=1 ./scripts/visualize.sh RGBNT201 video
set -euo pipefail

PROJECT_DIR="/home/buiminhly2303/Desktop/Master/nhandang/IDEA"
PYTHON="/home/buiminhly2303/miniconda3/envs/IDEA/bin/python"

# Rank-list video settings.
TOPK="${TOPK:-10}"
NUM_QUERIES="${NUM_QUERIES:-40}"
FPS="${FPS:-4}"
FEATS="${FEATS:-LOCAL}"
# CDA figure settings.
NUM_IMAGES="${NUM_IMAGES:-4}"
# Shared.
CKPT_NAME="${CKPT_NAME:-IDEAbest.pth}"
TEST_BATCH="${TEST_BATCH:-64}"

# Datasets that both visualizations support. RGBNT100 is deliberately excluded:
# there is no trained checkpoint in this tree and no image layout registered for it.
ALL_DATASETS=(RGBNT201 MSVR310)

DATASET_ARG="${1:-all}"
STEP_ARG="${2:-all}"

if [[ "${DATASET_ARG}" == "all" ]]; then
    DATASETS=("${ALL_DATASETS[@]}")
else
    DATASETS=("${DATASET_ARG}")
fi

if [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: python not found at ${PYTHON}" >&2
    echo "Activate the IDEA env or set PYTHON=/path/to/python" >&2
    exit 1
fi

cd "${PROJECT_DIR}"

for DATASET in "${DATASETS[@]}"; do
    CONFIG="configs/${DATASET}/IDEA.yml"
    OUT_DIR="IDEA_${DATASET}"
    WEIGHT="./${OUT_DIR}/${CKPT_NAME}"

    if [[ ! -f "${CONFIG}" ]]; then
        echo "SKIP ${DATASET}: no config at ${CONFIG}" >&2
        continue
    fi
    if [[ ! -f "${WEIGHT}" ]]; then
        echo "SKIP ${DATASET}: no checkpoint at ${WEIGHT}" >&2
        continue
    fi

    echo ""
    echo "=================================================================="
    echo " ${DATASET}  (weights: ${WEIGHT})"
    echo "=================================================================="

    if [[ "${STEP_ARG}" == "all" || "${STEP_ARG}" == "video" ]]; then
        echo "--- rank-list video: top-${TOPK}, ${NUM_QUERIES} queries, ${FPS} fps, feats=${FEATS} ---"
        "${PYTHON}" visualize_video.py \
            --config_file "${CONFIG}" \
            --topk "${TOPK}" \
            --num_queries "${NUM_QUERIES}" \
            --fps "${FPS}" \
            --feats "${FEATS}" \
            TEST.WEIGHT "${WEIGHT}" \
            TEST.IMS_PER_BATCH "${TEST_BATCH}"
    fi

    if [[ "${STEP_ARG}" == "all" || "${STEP_ARG}" == "maps" ]]; then
        echo "--- CDA offset / attention figures: ${NUM_IMAGES} images ---"
        "${PYTHON}" visualize_maps.py \
            --config_file "${CONFIG}" \
            --num_images "${NUM_IMAGES}" \
            TEST.WEIGHT "${WEIGHT}" \
            TEST.IMS_PER_BATCH "${TEST_BATCH}"
    fi
done

echo ""
echo "=================================================================="
echo " Done. Outputs:"
for DATASET in "${DATASETS[@]}"; do
    OUT_DIR="IDEA_${DATASET}"
    [[ -f "${OUT_DIR}/rank_video_${DATASET}.mp4" ]] && \
        echo "   ${OUT_DIR}/rank_video_${DATASET}.mp4"
    if [[ -d "${OUT_DIR}/cda_vis" ]]; then
        COUNT=$(find "${OUT_DIR}/cda_vis" -name '*.png' | wc -l)
        echo "   ${OUT_DIR}/cda_vis/  (${COUNT} figures)"
    fi
done
echo "=================================================================="
