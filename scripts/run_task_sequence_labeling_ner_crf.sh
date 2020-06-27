CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/pretrained_models/bert-base
export DATA_DIR=$CURRENT_DIR/dataset
export OUTPUR_DIR=$CURRENT_DIR/outputs
export TASK_NAME=cner

python task_sequence_labeling_ner_crf.py \
  --model_type=bert \
  --model_path=$BERT_BASE_DIR \
  --task_name=${TASK_NAME} \
  --do_train \
  --gpu=0 \
  --use_crf \
  --do_lower_case \
  --monitor=eval_f1 \
  --data_dir=$DATA_DIR/${TASK_NAME}/ \
  --train_max_seq_length=128 \
  --eval_max_seq_length=512 \
  --per_gpu_train_batch_size=24 \
  --per_gpu_eval_batch_size=24 \
  --learning_rate=3e-5 \
  --num_train_epochs=4.0 \
  --logging_steps=160 \
  --save_steps=160 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42

