CURRENT_DIR=`pwd`
export MODEL_DIR=$CURRENT_DIR/pretrained_models/bert-base-cn
export DATA_DIR=$CURRENT_DIR/dataset
export OUTPUR_DIR=$CURRENT_DIR/outputs
export TASK_NAME=ccks2021
export MODEL_TYPE=bert

python examples/task_text_similarity_ccks2021.py \
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
  --train_input_file=ccks2021_train_seed42_fold0.json \
  --eval_input_file=ccks2021_dev_seed42_fold0.json \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --train_max_seq_length=100 \
  --eval_max_seq_length=100 \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=32 \
  --learning_rate=3e-5 \
  --num_train_epochs=3 \
  --gradient_accumulation_steps=1 \
  --warmup_rate=0.1 \
  --scheduler_type=linear \
  --scheduler_on=batch \
  --weight_decay=0.01 \
  --logging_steps=-1 \
  --save_steps=-1 \
  --seed=42


