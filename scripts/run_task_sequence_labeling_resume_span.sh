CURRENT_DIR=`pwd`
export MODEL_DIR=$CURRENT_DIR/pretrained_models/bert-base-cn
export DATA_DIR=$CURRENT_DIR/dataset
export OUTPUR_DIR=$CURRENT_DIR/outputs
export TASK_NAME=resume
export MODEL_TYPE=bert

python examples/task_sequence_labeling_resume_span.py \
    --task_name=$TASK_NAME \
    --pretrained_model_path=$MODEL_DIR \
    --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
    --model_type=$MODEL_TYPE \
    --data_dir=dataset/$TASK_NAME/ \
    --do_train \
    --do_lower_case \
    --experiment_name=ver_001 \
    --checkpoint_save_best \
    --checkpoint_mode=max \
    --checkpoint_monitor=eval_f1_micro \
    --train_input_file=train.txt \
    --eval_input_file=dev.txt \
    --train_max_seq_length=200 \
    --eval_max_seq_length=512 \
    --test_max_seq_length=512 \
    --per_gpu_train_batch_size=24 \
    --per_gpu_eval_batch_size=24 \
    --per_gpu_test_batch_size=24 \
    --learning_rate=3e-5 \
    --num_train_epochs=6 \
    --gradient_accumulation_steps=1 \
    --warmup_rate=0.1 \
    --logging_steps=-1 \
    --save_steps=-1 \
    --seed=42