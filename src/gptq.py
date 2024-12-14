import argparse
import gc

import pandas as pd
import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-32B-Instruct")
parser.add_argument("--lora_weight_path", type=str, default="/workspace/weight")
parser.add_argument("--merged_model_path", type=str, default="/workspace/merged")
parser.add_argument("--output_dir", type=str, default="/workspace/gptq_weight")
parser.add_argument(
    "--data_path",
    type=str,
    default="/workspace/val_prompt.csv",
)


if __name__ == "__main__":
    args = parser.parse_args()

    # merge model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map="cpu", torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(model, args.lora_weight_path)
    model = model.merge_and_unload()
    model.save_pretrained(args.merged_model_path)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    max_len = 2048
    quantize_config = BaseQuantizeConfig(
        bits=4,
        group_size=32,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoGPTQForCausalLM.from_pretrained(
        args.merged_model_path, quantize_config, torch_dtype=torch.bfloat16
    ).to("cpu")

    data = []
    dataset = pd.read_csv(args.data_path).sample(n=128, random_state=42)["prompt"].tolist()
    for text in dataset:
        model_inputs = tokenizer([text])
        input_ids = torch.tensor(model_inputs.input_ids[:max_len], dtype=torch.int)
        data.append(dict(input_ids=input_ids, attention_mask=input_ids.ne(tokenizer.pad_token_id)))

    model.quantize(data, cache_examples_on_gpu=False)
    model.save_quantized(args.output_dir, use_safetensors=True)
    tokenizer.save_pretrained(args.output_dir)
