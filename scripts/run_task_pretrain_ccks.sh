CURRENT_DIR=`pwd`
export MODEL_DIR=$CURRENT_DIR/pretrained_models/macbert-base-cn
export DATA_DIR=$CURRENT_DIR/dataset
export OUTPUR_DIR=$CURRENT_DIR/outputs
export TASK_NAME=ccks2021
export MODEL_TYPE=bert

#-----------training-----------------
python examples/task_pretrain_ccks.py \
  --model_type=$MODEL_TYPE \
  --pretrained_model_path=$MODEL_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_lower_case \
  --device_id='0' \
  --checkpoint_mode=min \
  --experiment_name='ver_001' \
  --checkpoint_monitor=train_loss \
  --data_dir=$DATA_DIR/$TASK_NAME/ \
  --train_input_file=round1_train.txt \
  --eval_input_file=round2_train.txt \
  --test_input_file=Xeon3NLP_round1_test_20210524.txt \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --mlm_probability=0.15 \
  --train_max_seq_length=128 \
  --eval_max_seq_length=128 \
  --per_gpu_train_batch_size=16 \
  --learning_rate=1e-4 \
  --num_train_epochs=20 \
  --gradient_accumulation_steps=8 \
  --warmup_rate=0.1 \
  --scheduler_type=cosine \
  --logging_steps=1000 \
  --save_steps=4000 \
  --seed=42