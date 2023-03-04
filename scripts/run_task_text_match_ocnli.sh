CURRENT_DIR=`pwd`
export MODEL_DIR=$CURRENT_DIR/pretrained_models/macbert-base-cn
export DATA_DIR=$CURRENT_DIR/dataset
export OUTPUR_DIR=$CURRENT_DIR/outputs
export TASK_NAME=ocnli
export MODEL_TYPE=bert

#-----------training-----------------
python examples/task_text_match_ocnli.py \
  --model_type=$MODEL_TYPE \
  --pretrained_model_path=$MODEL_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_lower_case \
  --device_id='0' \
  --experiment_name='ver_001' \
  --checkpoint_mode=max \
  --checkpoint_monitor=eval_acc \
  --data_dir=$DATA_DIR/$TASK_NAME/ \
  --train_input_file=train.json \
  --eval_input_file=dev.json \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --train_max_seq_length=128 \
  --eval_max_seq_length=128 \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=32 \
  --learning_rate=3e-5 \
  --num_train_epochs=5 \
  --gradient_accumulation_steps=1 \
  --warmup_rate=0.1 \
  --scheduler_type=linear \
  --logging_steps=-1 \
  --save_steps=-1 \
  --seed=42





