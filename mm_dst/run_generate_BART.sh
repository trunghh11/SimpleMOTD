#!/bin/bash
if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
else
    PATH_DIR=$(realpath "$1")
fi

# Generate sentences (Furniture, multi-modal)
CUDA_VISIBLE_DEVICES=0 python -m gpt2_dst.scripts.run_generation_BART \
    --model_type=facebook/bart-base \
    --model_name_or_path="${PATH_DIR}"/gpt2_dst/save/model/ \
    --num_return_sequences=1 \
    --length=100 \
    --seed=1\
    --stop_token='<end>' \
    --prompts_from_file="${PATH_DIR}"/gpt2_dst/data/simmc2_dials_dstc10_devtest_predict.txt \
    --path_output="${PATH_DIR}"/gpt2_dst/results/simmc2_dials_dstc10_devtest_predicted.txt\
