#!/bin/bash
# This script is used to run local test on PASCAL VOC 2012 using MobileNet-v2.
# Users could also modify from this script for their use case.
#
# Usage:
#   sh ./local_test_mobilenetv2.sh

set -e

WORK_DIR=$(pwd)
cd ../models/research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
CURRENT_DIR=$(pwd)
DEEPLAB_WORK_DIR="${CURRENT_DIR}/deeplab"

python "${DEEPLAB_WORK_DIR}/model_test.py"
DATASET_DIR="datasets"
cd "${WORK_DIR}/${DATASET_DIR}"
sh download_and_convert_my_first_perception.sh


cd "${CURRENT_DIR}"
PASCAL_FOLDER="my_first_perception"
EXP_FOLDER="exp/train_on_mobilenetv2"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/export"
mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

TF_INIT_ROOT="https://storage.googleapis.com/mobilenet_v2/checkpoints"
CKPT_NAME="mobilenet_v2"
TF_INIT_CKPT="${CKPT_NAME}_1.0_224.tgz"
mkdir "${INIT_FOLDER}/${CKPT_NAME}"
cd "${INIT_FOLDER}/${CKPT_NAME}"
wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
tar -xf "${TF_INIT_CKPT}"
cd "${CURRENT_DIR}"
PASCAL_DATASET="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/tfrecord"

NUM_ITERATIONS=10
python "${DEEPLAB_WORK_DIR}"/train.py \
  --logtostderr \
  --train_split="train" \
  --model_variant="mobilenet_v2" \
  --output_stride=16 \
  --train_crop_size="513,513" \
  --train_batch_size=4 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=true \
  --tf_initial_checkpoint="${INIT_FOLDER}/${CKPT_NAME}/mobilenet_v2_1.0_224.ckpt" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${PASCAL_DATASET}" \
  --dataset="${PASCAL_FOLDER}"

python "${DEEPLAB_WORK_DIR}"/eval.py \
  --logtostderr \
  --eval_split="val" \
  --model_variant="mobilenet_v2" \
  --eval_crop_size="513,513" \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset_dir="${PASCAL_DATASET}" \
  --dataset="${PASCAL_FOLDER}" \
  --max_number_of_evaluations=1

python "${DEEPLAB_WORK_DIR}"/vis.py \
  --logtostderr \
  --vis_split="val" \
  --model_variant="mobilenet_v2" \
  --vis_crop_size="513,513" \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --vis_logdir="${VIS_LOGDIR}" \
  --dataset_dir="${PASCAL_DATASET}" \
  --dataset="${PASCAL_FOLDER}" \
  --max_number_of_iterations=1

CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"

python "${DEEPLAB_WORK_DIR}"/export_model.py \
  --logtostderr \
  --checkpoint_path="${CKPT_PATH}" \
  --export_path="${EXPORT_PATH}" \
  --model_variant="mobilenet_v2" \
  --num_classes=11 \
  --crop_size=513 \
  --crop_size=513 \
  --inference_scales=1.0
