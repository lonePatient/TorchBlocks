CURRENT_DIR=`pwd`
export MODEL_DIR=$CURRENT_DIR/pretrained_models/bert-base-en
export DATA_DIR=$CURRENT_DIR/dataset
export OUTPUR_DIR=$CURRENT_DIR/outputs
export TASK_NAME=cola
export MODEL_TYPE=bert

#-----------training-----------------
python task_text_classification_cola.py \
  --model_type=$MODEL_TYPE \
  --pretrained_model_path=$MODEL_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_lower_case \
  --fp16 \
  --device_id='0' \
  --experiment_code='V0' \
  --checkpoint_mode=max \
  --checkpoint_monitor=eval_mcc \
  --data_dir=$DATA_DIR/$TASK_NAME/ \
  --train_input_file=train.tsv \
  --eval_input_file=dev.tsv \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --train_max_seq_length=128 \
  --eval_max_seq_length=128 \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3 \
  --gradient_accumulation_steps=1 \
  --warmup_proportion=0.1 \
  --logging_steps=-1 \
  --save_steps=-1 \
  --seed=42



