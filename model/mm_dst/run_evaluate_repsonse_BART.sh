# Evaluate (Fashion, multi-modal)
CUDA_VISIBLE_DEVICES=0  python -m gpt2_dst.scripts.evaluate_response_BART \
    --input_path_target="./gpt2_dst/data/simmc2_dials_dstc10_devtest_target.txt" \
    --input_path_predicted="./gpt2_dst/results/simmc2_dials_dstc10_devtest_predicted.txt" \
    --output_path_report="./gpt2_dst/results/simmc2_dials_dstc10_devtest_response.txt"