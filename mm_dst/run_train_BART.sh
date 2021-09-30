#!/bin/bash
if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
else
    PATH_DIR=$(realpath "$1")
fi

# Train (multi-modal)
CUDA_VISIBLE_DEVICES=0 python3 -m gpt2_dst.scripts.run_language_modeling_BART \
    --output_dir="${PATH_DIR}"/gpt2_dst/save/model \
    --model_type=facebook/bart-base \
    --model_name_or_path=facebook/bart-base\
    --do_train\
    --line_by_line \
    --add_special_tokens="${PATH_DIR}"/gpt2_dst/data/simmc2_special_tokens.json \
    --train_data_pred_file="${PATH_DIR}"/gpt2_dst/data/simmc2_dials_dstc10_train_predict.txt \
    --train_data_target_file="${PATH_DIR}"/gpt2_dst/data/simmc2_dials_dstc10_train_target.txt \
    --do_eval\
    --eval_data_pred_file="${PATH_DIR}"/gpt2_dst/data/simmc2_dials_dstc10_dev_predict.txt \
    --eval_data_target_file="${PATH_DIR}"/gpt2_dst/data/simmc2_dials_dstc10_dev_target.txt \
    --num_train_epochs=10 \
    --overwrite_output_dir \
    --seed=1\
    --per_gpu_train_batch_size=12\
    --per_gpu_eval_batch_size=12\
    --save_steps=50000 \
    --logging_steps=50 \
    
