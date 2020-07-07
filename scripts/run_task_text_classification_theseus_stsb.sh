CURRENT_DIR=`pwd`
export MODEL_DIR=$CURRENT_DIR/pretrained_models/bert-base-en
export DATA_DIR=$CURRENT_DIR/dataset
export OUTPUR_DIR=$CURRENT_DIR/outputs
export TASK_NAME=sst-2

# ------------------ save every epoch two gpus--------------
python task_text_classification_theseus_sst2.py \
  --model_type=bert \
  --model_path=$MODEL_DIR \
  --predecessor_model_path=$OUTPUR_DIR/${TASK_NAME}_output/bert-base-en/checkpoint-best/ \
  --task_name=$TASK_NAME \
  --model_name=bert-theseus \
  --do_train \
  --do_lower_case \
  --gpu=0 \
  --do_save_best \
  --mcpt_mode=max \
  --monitor=eval_acc \
  --replacing_rate=0.3 \
  --scheduler_type=linear \
  --scheduler_linear_k=0.0006 \
  --data_dir=$DATA_DIR/${TASK_NAME}/ \
  --train_max_seq_length=128 \
  --eval_max_seq_length=128 \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=10.0 \
  --logging_steps=2105 \
  --save_steps=2105 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42
