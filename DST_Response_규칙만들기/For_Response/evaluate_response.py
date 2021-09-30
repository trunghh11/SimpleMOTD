#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the LICENSE file in the
root directory of this source tree.

    Scripts for evaluating the GPT-2 DST model predictions.

    First, we parse the line-by-line stringified format into responses
    and compute BLEU score.
"""
import argparse
import json
from gpt2_dst.utils.convert import parse_flattened_results_from_file
from utils.evaluate_dst import evaluate_from_flat_list

import nltk
import numpy as np


def normalize_sentence(sentence):
    """Normalize the sentences and tokenize."""
    return nltk.tokenize.word_tokenize(sentence.lower())


def parse_response_from_file(input_path):
    """Parses the response from a flattened file.

    Args:
        input_path: Path to read the responses from.
    """
    lines = []
    with open(input_path, "r") as file_id:
        for ii in file_id.readlines():
            split_line = ii.split("<EOB>", 1)
            if len(split_line) == 1:       # split_line 길이가 1이라는 건, EOB 토큰이 없다는 거고 생성 실패했다는 소리
                lines.append((split_line[0].strip("\n"), ""))
            else:
                lines.append(
                    (split_line[0].strip("\n"), split_line[1].strip("\n").strip("<EOS>").lstrip())
                )
#         a = split_line[0].strip("\n")
#         print("\n\nsplit_line[0] : {}".format(a))

#         a = split_line[1].strip("\n").strip("<EOS>").lstrip()
#         print("\nsplit_line[1] : {}".format(a))
        
    return lines


if __name__ == "__main__":
    # Parse input args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path_target", help="path for target, line-separated format (.txt)"
    )
    parser.add_argument(
        "--input_path_predicted",
        help="path for model prediction output, line-separated format (.txt)",
    )
    parser.add_argument(
        "--output_path_report", help="path for saving evaluation summary (.json)"
    )

    args = parser.parse_args()
    input_path_target = args.input_path_target
    input_path_predicted = args.input_path_predicted
    output_path_report = args.output_path_report

    # Convert the data from the GPT-2 friendly format to JSON
    list_target = parse_response_from_file(input_path_target)
    list_predicted = parse_response_from_file(input_path_predicted)

    # Compute BLEU scores.
    bleu_scores = []
    # Smoothing function.
    chencherry = nltk.translate.bleu_score.SmoothingFunction()

    for response, gt_response in zip(list_predicted, list_target):
#         print("\nPred : {}".format(len(list(response[0]))))
#         print("Generate : {}".format(len(list(gt_response[0]))))

#         pp = list()
#         gg = list()
#         for p, g in zip(list(response[0]), list(gt_response[0])):
#             if not p==g:
#                 pp.append(p)
#                 gg.append(g)
#         print("\nPP : {}".format(("").join(pp)))
#         print("GG : {}".format(("").join(gg)))
                    
#         assert response[0] == gt_response[0], "Input contexts do not match!"
        bleu_score = nltk.translate.bleu_score.sentence_bleu(
            [normalize_sentence(gt_response[1])],
            normalize_sentence(response[1]),
            smoothing_function=chencherry.method7,
        )
        bleu_scores.append(bleu_score)
    print(
        "BLEU score: {} +- {}".format(
            np.mean(bleu_scores), np.std(bleu_scores) / np.sqrt(len(bleu_scores))
        )
    )
