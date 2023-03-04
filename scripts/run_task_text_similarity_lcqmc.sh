CURRENT_DIR=`pwd`
export MODEL_DIR=$CURRENT_DIR/pretrained_models/bert-base-cn
export DATA_DIR=$CURRENT_DIR/dataset
export OUTPUR_DIR=$CURRENT_DIR/outputs
export TASK_NAME=lcqmc
export MODEL_TYPE=bert

#-----------training-----------------
python examples/task_text_similarity_lcqmc.py \
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
  --train_input_file=train.txt \
  --eval_input_file=dev.txt \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --train_max_seq_length=50 \
  --eval_max_seq_length=50 \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=3 \
  --gradient_accumulation_steps=1 \
  --warmup_rate=0.1 \
  --scheduler_type=linear \
  --scheduler_on=batch \
  --logging_steps=-1 \
  --save_steps=-1 \
  --seed=42