from pipeline.model_utils.model_base import ModelBase

def construct_model_base(model_path: str, **kwargs) -> ModelBase: # Added **kwargs
    if 'llama-3' in model_path.lower():
        from pipeline.model_utils.llama3_model import Llama3Model
        return Llama3Model(model_path, **kwargs)
    elif 'gemma' in model_path.lower():
        from pipeline.model_utils.gemma3_model import GemmaModel
        return GemmaModel(model_path, **kwargs)
    elif "qwen" in model_path.lower():
        from pipeline.model_utils.qwen_model import Qwen3Model
        return Qwen3Model(model_path, **kwargs) # Passes kwargs to Qwen3Model
    else:
        raise ValueError(f"Unknown model family: {model_path}")