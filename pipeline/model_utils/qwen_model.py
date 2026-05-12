import torch
from torch import Tensor
from jaxtyping import Float
from transformers import AutoTokenizer, AutoModelForCausalLM
from pipeline.model_utils.model_base import ModelBase

class Qwen3Model(ModelBase):
    def __init__(self, model_name_or_path: str, **kwargs):
        # Pass everything to ModelBase.__init__
        super().__init__(model_name_or_path, **kwargs)

    def _load_model(self, model_name_or_path: str, **kwargs) -> AutoModelForCausalLM:
        """Loads model with support for 4-bit quantization and bfloat16."""
        # Separate 'enable_thinking' if it's not a standard HF argument
        kwargs.pop('enable_thinking', None) 
        
        return AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            **kwargs # This now contains load_in_4bit=True
        )

    def _load_tokenizer(self, model_name_or_path: str) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        return tokenizer

    def _get_tokenize_instructions_fn(self):
        def tokenize_instructions(instructions):
            prompts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": instr}], 
                    tokenize=False, 
                    add_generation_prompt=True
                ) for instr in instructions
            ]
            return self.tokenizer(prompts, padding=True, return_tensors="pt")
        return tokenize_instructions

    def _get_eoi_toks(self):
        # Targets the end of the assistant header in Qwen3
        return [self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)[-1]]

    def _get_thresholds(self):
        return {"default": 0.5}

    def _get_unanswerability_toks(self):
        return self.tokenizer.encode("I cannot answer", add_special_tokens=False)

    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_attn_modules(self):
        return [layer.self_attn for layer in self.model.model.layers]

    def _get_mlp_modules(self):
        return [layer.mlp for layer in self.model.model.layers]

    def _get_act_add_mod_fn(self, direction, coeff, layer):
        def hook(module, input, output):
            h = output[0]
            h += coeff * direction.to(h.device)
            return (h,) + output[1:]
        return hook