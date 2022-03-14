#!/bin/bash
# Script to download and preprocess a Unity Perception dataset.
#
# Usage:
#   bash ./download_and_convert_my_first_perception.sh
#
# The folder structure is assumed to be:
#  + datasets
#     - download_and_convert_my_first_perception.sh
#     + my_first_perception
#       + MyFirstPerception
#         + train
#           + Dataset<GUID>
#           + Logs
#           + RGB<GUID>
#           + SemanticSegmentation<GUID>
#         + val
#           + Dataset<GUID>
#           + Logs
#           + RGB<GUID>
#           + SemanticSegmentation<GUID>
set -e

CURRENT_DIR=$(pwd)
cd ../..
DEEPLAB_SCRIPT_DIR="$(pwd)/models/research/deeplab/datasets"
cd "${CURRENT_DIR}"
WORK_DIR="./my_first_perception"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

download_and_uncompress() {
  local FILE_ID="${1}"
  local FILENAME=${2}

  if [ ! -f "${FILENAME}" ]; then
    echo "Downloading ${FILENAME} to ${WORK_DIR}"
    curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=t&id=${FILE_ID}" -o "${FILENAME}"
  fi
  echo "Uncompressing ${FILENAME}"
  tar xzvf "${FILENAME}"
}

download_and_uncompress "${UNITY_DATASET_FILE_ID}" "${UNITY_DATASET_FILE_NAME}"
cd "${CURRENT_DIR}"
PASCAL_ROOT="${WORK_DIR}/${UNITY_DATASET_FILE_NAME%%.*}"
python "${SCRIPT_DIR}/convert_my_first_perception_data.py" \
        "${PASCAL_ROOT}" \
        --output_dir="${WORK_DIR}"

SEMANTIC_SEG_FOLDER="${PASCAL_ROOT}/SegmentationClass"
OUTPUT_DIR="${WORK_DIR}/tfrecord"
mkdir -p "${OUTPUT_DIR}"

IMAGE_FOLDER="${PASCAL_ROOT}/JPEGImages"
LIST_FOLDER="${PASCAL_ROOT}/ImageSets"

echo "Converting a Unity Perception dataset..."
python "${DEEPLAB_SCRIPT_DIR}/build_voc2012_data.py" \
  --image_folder="${IMAGE_FOLDER}" \
  --semantic_segmentation_folder="${SEMANTIC_SEG_FOLDER}" \
  --list_folder="${LIST_FOLDER}" \
  --image_format="jpg" \
  --output_dir="${OUTPUT_DIR}"
