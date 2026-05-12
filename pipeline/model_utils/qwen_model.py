import torch
import functools
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List
from torch import Tensor
from jaxtyping import Float
import numpy as np

from pipeline.model_utils.model_base import ModelBase

# Qwen3 (Im-style) template focused on classification
QWEN3_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

# The instruction we will inject to force binary classification
# Updated to use the single-token 'multiple'
ENUM_PROMPT_PREFIX = "Does the following question have multiple possible answers? Answer with exactly one word: 'multiple' if yes, 'singular' if no.\n\nQuestion: "

# Using your verified ID for 'multiple'
QWEN3_ENUMERABILITY_TOKS = [35673]

def format_instruction_qwen3(instruction: str):
    # We wrap the question with our classification command
    full_instruction = ENUM_PROMPT_PREFIX + instruction
    return QWEN3_CHAT_TEMPLATE.format(instruction=full_instruction)

def tokenize_instructions_qwen3(tokenizer: AutoTokenizer, instructions: List[str]):
    prompts = [format_instruction_qwen3(instr) for instr in instructions]
    return tokenizer(prompts, padding=True, truncation=True, max_length=256, return_tensors="pt")

class Qwen3Model(ModelBase):
    def __init__(self, model_name_or_path: str, **kwargs):
        super().__init__(model_name_or_path, **kwargs)

    def _load_model(self, model_name_or_path: str, **kwargs) -> AutoModelForCausalLM:
        # Handle quantization config explicitly as we discussed for OOM
        if kwargs.get('load_in_4bit'):
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            kwargs.pop('load_in_4bit')

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            **kwargs
        ).eval()
        model.requires_grad_(False)
        return model

    def _load_tokenizer(self, model_name_or_path: str) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_qwen3, tokenizer=self.tokenizer)

    def _get_eoi_toks(self):
        # This targets the newline after assistant header
        return self.tokenizer.encode("\n", add_special_tokens=False)

    def _get_thresholds(self):
        return [float(x) for x in np.arange(-1, 1.1, 0.1)]

    def _get_unanswerability_toks(self):
        return QWEN3_ENUMERABILITY_TOKS

    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList([block.self_attn for block in self.model.model.layers])

    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block.mlp for block in self.model.model.layers])

    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        """Standard hook logic for activation steering."""
        def hook(module, input, output):
            h = output[0]
            h += coeff * direction.to(h.device)
            return (h,) + output[1:]
        return hook