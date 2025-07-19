# Monkey patch to handle torchvision import error
import sys
from unittest.mock import Mock

class MockTorchvision:
    def __getattr__(self, name):
        return Mock()

# Temporarily replace torchvision to prevent the error
original_import = __builtins__.__import__

def patched_import(name, *args, **kwargs):
    if name == 'torchvision' or name.startswith('torchvision.'):
        return MockTorchvision()
    return original_import(name, *args, **kwargs)

__builtins__.__import__ = patched_import

import torch
from attention_patch import replace_attention_mask

replace_attention_mask()

try:
    from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM
finally:
    # Restore original import
    __builtins__.__import__ = original_import

import torch.nn.functional as F
from argparse import ArgumentParser


from model import DiscreteDiffusionModel, generate_samples

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default='myduy/test-diffusionlm-7b-v1')
    parser.add_argument("--shift", type=bool, default=True) # do not change this
    parser.add_argument("--diffusion_steps", type=int, default=64)
    parser.add_argument("--logits_temp", type=float, default=0.9)
    parser.add_argument("--topp_temp", type=float, default=0.9)
    parser.add_argument("--verbose", type=bool, default=False) # print middle state
    parser.add_argument("--flash_attn", type=str, choices=["eager", "sdpa", "flash_attention_2"], default="eager") # print middle state

    args = parser.parse_args()

    # model_name = 'gpt2'  # 'gpt2-medium', 'gpt2-large'
    model_name = args.model_name
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        _attn_implementation=args.flash_attn, 
        torch_dtype=torch.bfloat16
    )
    # model = DiscreteDiffusionModel(args.base_model_name, config, tokenizer)

    model = DiscreteDiffusionModel(
        model=model, 
        config=config, 
        tokenizer=tokenizer,
        device='cuda'
    ).to('cuda')

    # import pdb; pdb.set_trace();

    gen_len = 256
    print("="*20, "Generating...", gen_len)
    # un-conditional generation
    print("="*20, "Unconditional gen...")
    x0 = [0] * gen_len
    inputs = {
        "input_ids": torch.tensor([x0])
    }
    res = generate_samples(model, args, tokenizer, inputs, verbose=args.verbose)
    pred = tokenizer.decode(res.tolist()[0])
    print(pred)

    # conditional generation
    print("="*20, "Prefix gen...")
    instruction = "<ner>\nTiêm chủng vaccine ngừa virus SARS-CoV-2 là biện pháp hiệu quả.\n</ner>"
    prefix = [tokenizer.bos_token_id] + tokenizer.encode(instruction)

    src_mask = [1]*len(prefix)+[0]*(gen_len-len(prefix))
    x0 = prefix + [0]*(gen_len-len(prefix))

    inputs = {
        "input_ids": torch.tensor([x0]), 
        "src_mask": torch.tensor([src_mask])
    }
    res = generate_samples(model, args, tokenizer, inputs, verbose=args.verbose)
    pred = tokenizer.decode(res.tolist()[0])
    print(pred)