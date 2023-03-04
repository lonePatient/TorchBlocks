CURRENT_DIR=`pwd`
export MODEL_DIR=$CURRENT_DIR/pretrained_models/bert-base-cn
export DATA_DIR=$CURRENT_DIR/dataset
export OUTPUR_DIR=$CURRENT_DIR/outputs
export TASK_NAME=resume
export MODEL_TYPE=bert

#-----------training-----------------
python examples/task_sequence_labeling_resume_crf.py \
  --model_type=$MODEL_TYPE \
  --pretrained_model_path=$MODEL_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_lower_case \
  --device_id='0' \
  --experiment_name='ver_001' \
  --checkpoint_mode=max \
  --checkpoint_monitor=eval_f1_micro \
  --data_dir=$DATA_DIR/$TASK_NAME/ \
  --train_input_file=train.txt \
  --eval_input_file=dev.txt \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --train_max_seq_length=200 \
  --eval_max_seq_length=512 \
  --test_max_seq_length=512 \
  --per_gpu_train_batch_size=24 \
  --per_gpu_eval_batch_size=24 \
  --per_gpu_test_batch_size=24 \
  --learning_rate=2e-5 \
  --other_learning_rate=1e-3 \
  --num_train_epochs=20 \
  --gradient_accumulation_steps=1 \
  --warmup_rate=0.1 \
  --logging_steps=-1 \
  --save_steps=-1 \
  --seed=42



