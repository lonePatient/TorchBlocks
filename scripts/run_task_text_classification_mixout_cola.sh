CURRENT_DIR=`pwd`
export MODEL_DIR=$CURRENT_DIR/pretrained_models/bert-base-en
export DATA_DIR=$CURRENT_DIR/dataset
export OUTPUR_DIR=$CURRENT_DIR/outputs
export TASK_NAME=cola

# ------------------ save every epoch two gpus--------------
python task_text_classification_mixout_cola.py \
  --model_type=bert \
  --model_path=$MODEL_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_lower_case \
  --gpu=0 \
  --mixout=0.3 \
  --monitor=eval_mcc \
  --data_dir=$DATA_DIR/${TASK_NAME}/ \
  --train_max_seq_length=128 \
  --eval_max_seq_length=128 \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --logging_steps=268 \
  --save_steps=268 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42
