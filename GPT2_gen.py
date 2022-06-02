import argparse
import logging
import csv
import re

import numpy as np
import torch

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
)
MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

def mask_tokens(inputs, tokenizer, args):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    masked_indices = torch.bernoulli(torch.full(labels.shape, args.mlm_probability)).bool()
    labels[~masked_indices] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def prepare_ctrl_input(args, _, tokenizer, prompt_text):
    if args.temperature > 0.7:
        logger.info("CTRL typically works better with lower temperatures (and lower top_k).")

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
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
                language = input("Using XLM. Select language in " + str(list(available_languages)) + " >>> ")

        model.config.lang_id = model.config.lang2id[language]
        # kwargs["language"] = tokenizer.lang2id[language]

    # TODO fix mask_token_id setup when configurations will be synchronized between models and tokenizers
    # XLM masked-language modeling (MLM) models need masked token
    # is_xlm_mlm = "mlm" in args.model_name_or_path
    # if is_xlm_mlm:
    #     kwargs["mask_token_id"] = tokenizer.mask_token_id

    return prompt_text


def prepare_xlnet_input(args, _, tokenizer, prompt_text):
    prefix = args.prefix if args.prefix else args.padding_text if args.padding_text else PREFIX
    prompt_text = prefix + prompt_text
    return prompt_text


def prepare_transfoxl_input(args, _, tokenizer, prompt_text):
    prefix = args.prefix if args.prefix else args.padding_text if args.padding_text else PREFIX
    prompt_text = prefix + prompt_text
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

class GPT_2_gen(object):
    def __init__(self, args):
        args.device = torch.device(
            "cuda:" + str(args.device_id) if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

        # Initialize the model and tokenizer

        try:
            args.model_type = args.model_type.lower()
            self.model_class, self.tokenizer_class = MODEL_CLASSES[args.model_type]
        except KeyError:
            raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

        # set up model and tokenizer:
        self.tokenizer = self.tokenizer_class.from_pretrained(args.model_name_or_path)
        self.model = self.model_class.from_pretrained(args.model_name_or_path)
        self.model.to(args.device)

        # set args
        self.args = args
        self.args.length = adjust_length_to_model(args.length, max_sequence_length=self.model.config.max_position_embeddings)

    def gen_text(self, prompt_text, order_remain, output_obj=False, restriction=None):
        """
        Generate tokens with prompt
        :param prompt_text:
        :param order_remain: Which sentence you want to mak up.
        :return: the sentence in the order_remain order.
        """
        while True:
            # Different models need different input formatting and/or extra arguments
            requires_preprocessing = self.args.model_type in PREPROCESSING_FUNCTIONS.keys()
            if requires_preprocessing:
                prepare_input = PREPROCESSING_FUNCTIONS.get(self.args.model_type)
                preprocessed_prompt_text = prepare_input(self.args, self.model, self.tokenizer, prompt_text)

                if self.model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
                    tokenizer_kwargs = {"add_space_before_punct_symbol": True}
                else:
                    tokenizer_kwargs = {}

                encoded_prompt = self.tokenizer.encode(
                    preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt", **tokenizer_kwargs
                )
            else:
                prefix = self.args.prefix if self.args.prefix else self.args.padding_text
                encoded_prompt = self.tokenizer.encode(prefix + prompt_text, add_special_tokens=False, return_tensors="pt")
            encoded_prompt = encoded_prompt.to(self.args.device)

            if encoded_prompt.size()[-1] == 0:
                input_ids = None
            else:
                input_ids = encoded_prompt

            output_sequences = self.model.generate(
                input_ids=input_ids,
                max_length=self.args.length + len(encoded_prompt[0]),
                temperature=self.args.temperature,
                top_k=self.args.k,
                top_p=self.args.p,
                repetition_penalty=self.args.repetition_penalty,
                do_sample=True,
                num_return_sequences=5  # args.num_return_sequences
            )

            # Remove the batch dimension when returning multiple sequences
            if len(output_sequences.shape) > 2:
                output_sequences.squeeze_()

            for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
                # print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
                generated_sequence = generated_sequence.tolist()

                # Decode text
                text = self.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

                # Remove all text after the stop token
                text = text[: text.find(self.args.stop_token) if self.args.stop_token else None]

                #####BBB#####
                text = text.replace("\n", ".")
                text = text.replace(":", ".")
                text = text.replace("?", ".")
                text = text.replace("!", ".")

                if output_obj:
                    text = text.split(".")[order_remain]
                    prompt_text_len =len(prompt_text.split('.')[order_remain])
                    text = text[prompt_text_len+1:]

                else:
                    text = text.split(".")[order_remain] + '.'  # generated sentence

                if not restriction:
                    return text

    def gen_multiple_obj(self, prompt_text, num=1):
        """
        Generate multiple obj at once.
        :param num: # of obj you need
        :return: list of str.
        """
        res = []
        order_remain = prompt_text.count('.')
        for seed in range(self.args.seed, self.args.seed + num):
            res.append(self.gen_text(prompt_text=prompt_text,
                                     order_remain=order_remain,
                                     output_obj=True))
        return res

    def calculate_prob(self, sentences=[]):
        outputs = []
        for sentence in sentences:
            inputs = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sentence))
            inputs = torch.tensor(inputs)
            # inputs = self.tokenizer.add_special_tokens_single_sentence(inputs)
            labels = inputs.clone()
            inputs = inputs.to(self.args.device)
            labels = labels.to(self.args.device)
            output = float(self.model(inputs, labels=labels)[0].cpu().detach().numpy())
            print('The prob of ' + sentence + ' is ===> ', output)
            outputs.append(output)
        return outputs

    def set_args(self):
        parser_gpt = argparse.ArgumentParser()
        parser_gpt.add_argument(
            "--model_type",
            default='gpt2',
            type=str,
            required=False,
            help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
        )
        parser_gpt.add_argument(
            "--model_name_or_path",
            default='gpt2',
            type=str,
            required=False,
            help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
        )

        parser_gpt.add_argument("--prompt", type=str, default="")
        parser_gpt.add_argument("--length", type=int, default=20)
        parser_gpt.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

        parser_gpt.add_argument(
            "--temperature",
            type=float,
            default=1.0,
            help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
        )
        parser_gpt.add_argument(
            "--repetition_penalty", type=float, default=1.0,
            help="primarily useful for CTRL model; in that case, use 1.2"
        )
        parser_gpt.add_argument("--k", type=int, default=0)
        parser_gpt.add_argument("--p", type=float, default=0.9)

        parser_gpt.add_argument("--prefix", type=str, default="", help="Text added prior to input.")
        parser_gpt.add_argument("--padding_text", type=str, default="",
                            help="Deprecated, the use of `--prefix` is preferred.")
        parser_gpt.add_argument("--xlm_language", type=str, default="",
                            help="Optional language when used with the XLM model.")

        parser_gpt.add_argument("--seed", type=int, default=42, help="random seed for initialization")
        parser_gpt.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
        parser_gpt.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
        parser_gpt.add_argument(
            "--fp16",
            action="store_true",
            help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
        )
        parser_gpt.add_argument("--device_id",
                            type=int,
                            default=0,
                            help="gpu id")

        args_gpt = parser_gpt.parse_args()
        args_gpt.device = torch.device(
            "cuda:" + str(args_gpt.device_id) if torch.cuda.is_available() and not args_gpt.no_cuda else "cpu")
        args_gpt.n_gpu = 0 if args_gpt.no_cuda else torch.cuda.device_count()

        set_seed(args_gpt)
        self.args = args_gpt



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default='gpt2',
        type=str,
        required=False,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default='gpt2',
        type=str,
        required=False,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument("--prefix", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--padding_text", type=str, default="", help="Deprecated, the use of `--prefix` is preferred.")
    parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument("--device_id",
                        type=int,
                        default=0,
                        help="gpu id")

    args = parser.parse_args()
    args.device = torch.device(
        "cuda:" + str(args.device_id) if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    set_seed(args)

    gpt_2_gen = GPT_2_gen()
    # output = gpt_2_gen.gen_multiple_obj(prompt_text='Jenny lived in Florida.", "Jenny hears',
    #                                     num=3)
    output = gpt_2_gen.calculate_prob(sentences=['You are in the kitchen', 'You have kitchen.'])
    print('output => ', output)

