CURRENT_DIR=`pwd`
export MODEL_DIR=$CURRENT_DIR/pretrained_models/bert-base
export DATA_DIR=$CURRENT_DIR/dataset
export OUTPUR_DIR=$CURRENT_DIR/outputs
export TASK_NAME=afqmc

# ------------------ save every epoch --------------
python task_siamese_similarity_afqmc.py \
  --model_type=bert \
  --model_path=$MODEL_DIR \
  --task_name=${TASK_NAME} \
  --do_train \
  --gpu=0,1 \
  --do_lower_case \
  --monitor=eval_acc \
  --data_dir=$DATA_DIR/${TASK_NAME}/ \
  --train_max_seq_length=128 \
  --eval_max_seq_length=128 \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --logging_steps=80 \
  --save_steps=80 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42
