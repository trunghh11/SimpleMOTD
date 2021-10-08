#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)

Adapted from
https://github.com/huggingface/transformers/blob/master/examples/text-generation/run_generation.py
"""

import argparse
import logging
import os
import numpy as np
import random
import torch
from transformers import BartForConditionalGeneration as BartLMHeadModel
from transformers import T5ForConditionalGeneration as T5LMHeadModel

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, AutoTokenizer),
    "gpt2-medium": (GPT2LMHeadModel, AutoTokenizer),
    "gpt2-large": (GPT2LMHeadModel, AutoTokenizer),    
    "microsoft/dialogpt-small": (GPT2LMHeadModel, AutoTokenizer), 
    "t5-small": (T5LMHeadModel, AutoTokenizer), 
    "t5-base": (T5LMHeadModel, AutoTokenizer), 
    "t5-large": (T5LMHeadModel, AutoTokenizer), 
    "facebook/bart-base": (BartLMHeadModel, AutoTokenizer), 
    "facebook/bart-large": (BartLMHeadModel, AutoTokenizer), 
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_device(args):
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        args.device = torch.device('cpu')
        print('Device name: cpu')


#
# Functions to prepare models' input
#


def prepare_ctrl_input(args, _, tokenizer, prompt_text):
    if args.temperature > 0.7:
        logger.info(
            "CTRL typically works better with lower temperatures (and lower top_k)."
        )

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=True)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info(
            "WARNING! You are not starting your generation from a control code so you won't get good results"
        )
    return prompt_text


def prepare_xlm_input(args, model, tokenizer, prompt_text):
    # kwargs = {"language": None, "mask_token_id": None}

    # Set the language
    use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
    if hasattr(model.config, "lang2id") and use_lang_emb:
        available_languages = model.config.lang2id.keys()
        if args.xlm_language in available_languages:
            language = args.xlm_language
        else:
            language = None
            while language not in available_languages:
                language = input(
                    "Using XLM. Select language in "
                    + str(list(available_languages))
                    + " >>> "
                )

        model.config.lang_id = model.config.lang2id[language]
        # kwargs["language"] = tokenizer.lang2id[language]

    # TODO fix mask_token_id setup when configurations will be synchronized between models and tokenizers
    # XLM masked-language modeling (MLM) models need masked token
    # is_xlm_mlm = "mlm" in args.model_name_or_path
    # if is_xlm_mlm:
    #     kwargs["mask_token_id"] = tokenizer.mask_token_id

    return prompt_text


def prepare_xlnet_input(args, _, tokenizer, prompt_text):
    prompt_text = (
        args.padding_text if args.padding_text else PADDING_TEXT
    ) + prompt_text
    return prompt_text


def prepare_transfoxl_input(args, _, tokenizer, prompt_text):
    prompt_text = (
        args.padding_text if args.padding_text else PADDING_TEXT
    ) + prompt_text
    return prompt_text


PREPROCESSING_FUNCTIONS = {
    "ctrl": prepare_ctrl_input,
    "xlm": prepare_xlm_input,
    "xlnet": prepare_xlnet_input,
    "transfo-xl": prepare_transfoxl_input,
}


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument(
        "--prompts_from_file",
        type=str,
        default=None,
        help="""
read prompts from a file and generate, overrides any prompt given on the
command line""",
    )
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument(
        "--stop_token",
        type=str,
        default=None,
        help="Token at which text generation is stopped",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="primarily useful for CTRL model; in that case, use 1.2",
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument(
        "--padding_text",
        type=str,
        default="",
        help="Padding text for Transfo-XL and XLNet.",
    )
    parser.add_argument(
        "--xlm_language",
        type=str,
        default="",
        help="Optional language when used with the XLM model.",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="The number of samples to generate.",
    )
    parser.add_argument(
        "--path_output",
        type=str,
        default=None,
        help="Path to output predictions in a line separated text file.",
    )
    args = parser.parse_args()
    set_seed(args)

    args.device = set_device(args)

    if args.prompts_from_file and not os.path.exists(args.prompts_from_file):
        raise Exception(f"prompt file '{args.prompts_from_file}' not found")

    # Initialize the model and tokenizer
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError(
            "the model {} you specified is not supported. You are welcome to add it and open a PR :)"
        )

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)    
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    if 't5' in args.model_type:
        args.length = adjust_length_to_model(
            args.length, max_sequence_length=model.config.n_positions
        )
    else:
        args.length = adjust_length_to_model(
            args.length, max_sequence_length=model.config.max_position_embeddings
        )

    logger.info(args)

    results = []
    prompts = []
    if args.prompts_from_file:
        with open(args.prompts_from_file) as handle:
            prompts = handle.readlines()

    # print("\ntokenizer : {}".format(len(tokenizer)))
    # for token, ids in tokenizer.vocab.items():
    #     if ids >= 50260:
    #         print("{} : {}".format(token, ids))
    # import sys
    # sys.exit()
    while True:
        if not prompts:
            prompts = [args.prompt if args.prompt else input("Model prompt >>> ")]
            if not args.prompt and (
                len(prompts) == 0
                or prompts[0].strip() == ""
                or prompts[0].lower() == "quit"
            ):
                break  # break while True loop

        n_prompts = len(prompts)
        for i, prompt_text in enumerate(prompts):
            # if 5101 > i:
            #     print("{} 번째 Text : Pass ({})".format(i, len(prompt_text)))
            #     continue
            # print("\n{} 번째 Text ({}) : ".format(i, len(prompt_text)))
            # print("prompt_text : {}".format(prompt_text))            

            # Strip any trailing \n if provided
            prompt_text = prompt_text.strip("\n")

            # Different models need different input formatting and/or extra arguments
            requires_preprocessing = args.model_type in PREPROCESSING_FUNCTIONS.keys()
            if requires_preprocessing:
                prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_type)
                preprocessed_prompt_text = prepare_input(
                    args, model, tokenizer, prompt_text
                )
                encoded_prompt = tokenizer.encode(
                    preprocessed_prompt_text,
                    add_special_tokens=True,
                    return_tensors="pt",
                    max_length=tokenizer.model_max_length,
                    add_space_before_punct_symbol=True,
                )
            else:
                # print("ElSE 문")
                # encoded_prompt = tokenizer.encode(
                #     prompt_text, add_special_tokens=True, return_tensors="pt"
                # )
                encoded_prompt = tokenizer(
                    [prompt_text], 
                    max_length=tokenizer.model_max_length,                    
                    add_special_tokens=True)
            src = torch.tensor(encoded_prompt.input_ids).to(args.device)
            src_mask = torch.tensor(encoded_prompt.attention_mask).to(args.device)
            # if i > 515:
            #     # print("Max len : {}".format(tokenizer.model_max_length))
            #     # print("SRC : {}".format(encoded_prompt.input_ids))
            #     print("\nSRC : {}".format(src.size()))
            try:
                output_sequences = model.generate(
                    src,
                    max_length=args.length + len(src),
                    decoder_start_token_id=tokenizer.pad_token_id,
                    attention_mask=src_mask,
                    early_stopping=True)
            except Exception as e:
                print("\n\nError : {}".format(e))
                print("src type : {}".format(type(src)))
                print("src dim : {}".format(src.size()))
                print("src : {}".format(src[0]))

                print("src_mask type : {}".format(type(src_mask)))
                print("src_mask dim : {}".format(src_mask.size()))
                print("src_mask : {}".format(src_mask[0]))

                print("max_length : {}".format(args.length + len(src)))

                import sys
                sys.exit()
            # print("output_sequences : {} {}".format(output_sequences.size(), output_sequences))

            # Remove the batch dimension when returning multiple sequences
            if len(output_sequences.shape) > 2:
                output_sequences.squeeze_()

            generated_sequences = []

            for generated_sequence_idx, generated_sequence in enumerate(
                output_sequences
            ):
                print(
                    "=== GENERATED SEQUENCE {sequence_idx}, {promt_idx}/{n_prompts} ===".format(
                        sequence_idx=generated_sequence_idx + 1,
                        promt_idx=i + 1,
                        n_prompts=n_prompts,
                    )
                )
                generated_sequence = generated_sequence.tolist()

                # Decode text
                text = tokenizer.decode(
                    generated_sequence, clean_up_tokenization_spaces=True
                )
                # print("1 TEXT : {}".format(text))
                # print("stop_token : {}".format(args.stop_token))
                # print("SRC : {}".format(src))
                # Remove all text after the stop token
                text = text[: text.find(args.stop_token) if args.stop_token else None]
                text =text.replace("<pad><s>", "")

                print("2 TEXT : {}".format(text))

                # Add the prompt at the beginning of the sequence. Remove the
                # excess text that was used for pre-processing
                # total_sequence = (
                #     prompt_text
                #     + text[
                #         len(
                #             tokenizer.decode(
                #                 src[0], clean_up_tokenization_spaces=True
                #             )
                #         ) :
                #     ]
                # )

                generated_sequences.append(text)
            results.append(generated_sequences)
        prompts = []
        if args.prompt or args.prompts_from_file:
            break  # break while True loop

    if args.path_output is not None:

        # Create a directory if it does not exist
        directory = os.path.dirname(args.path_output)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # Format results into a line-separated string file
        str_results = "\n".join(
            [" || ".join(generated_sequences) for generated_sequences in results]
        )
        # Save to a file
        with open(args.path_output, "w") as f_out:
            f_out.write(str_results)

    return results


if __name__ == "__main__":
    main()
