import warnings
import re

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_tokenizer(model_name_or_path):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    return tokenizer

def load_model(model_name_or_path, quantization_config):
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map=None,
        trust_remote_code=True
    )
    
    model.eval()
    
    return model

def generate(turns, tokenizer, model, generation_config):

    input_ids = tokenizer.apply_chat_template(
        turns,
        add_generation_prompt=True,
        return_tensors='pt'
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            generation_config=generation_config
        )
        
    answer_tokens = outputs[0, input_ids.shape[1]:-1]
    return tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

def extract_block(text, block_type="md"):
    
    blocks = re.findall(r"```(?:{block_type})(.*?)```".format(block_type=block_type), text, re.DOTALL)
    
    if len(blocks) == 0:
        warnings.warn(f"Not found {block_type} block in:\n\n {text}".format(block_type=block_type, text=text))
        return ""
    
    if len(blocks) > 1:
        warnings.warn(f"Found multiple {block_type} block in:\n\n {text}".format(block_type=block_type, text=text))
    
    return blocks[-1].strip()