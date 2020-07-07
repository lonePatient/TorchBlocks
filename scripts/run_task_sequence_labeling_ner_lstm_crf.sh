CURRENT_DIR=`pwd`
export MODEL_BASE_DIR=$CURRENT_DIR/pretrained_models/lstm-crf
export DATA_DIR=$CURRENT_DIR/dataset
export OUTPUR_DIR=$CURRENT_DIR/outputs
export TASK_NAME=cluener

python task_sequence_labeling_ner_lstm_crf.py \
  --model_type=lstm-crf \
  --model_path=$MODEL_BASE_DIR \
  --task_name=${TASK_NAME} \
  --do_train \
  --gpu=1 \
  --use_crf \
  --do_lower_case \
  --monitor=eval_f1 \
  --data_dir=$DATA_DIR/${TASK_NAME}/ \
  --train_max_seq_length=128 \
  --eval_max_seq_length=512 \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=32 \
  --learning_rate=0.001 \
  --num_train_epochs=50 \
  --max_grad_norm=5.0 \
  --logging_steps=336 \
  --save_steps=336 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42

