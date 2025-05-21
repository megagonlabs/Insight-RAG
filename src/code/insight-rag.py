from functions import functions_dict
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import login
from keys import API_KEYS
import numpy as np
import torch
import openai
import json
import random
import sys
import csv
import argparse
import logging
import os


HF_TOKEN = API_KEYS["hf_token"]        
openai.api_key = API_KEYS["openai-key"]        


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
root = logging.getLogger()       
root.setLevel(logging.WARNING)   
logger.setLevel(logging.INFO)    


def read_jsonl(path):
    test_data = []
    with open(path, 'r') as f:
        for line in f:
            dict_temp = json.loads(line)
            test_data += [dict_temp]
    return test_data


def save_jsonl(path, out):
    with open(path, 'w') as outfile:
        for entry in out:
            json.dump(entry, outfile)
            outfile.write('\n')
    return 


def check_path(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return


def insight_rag(main_model, insight_model, tokenizer, device, path_data, top_k, task_type):
    data = read_jsonl(path_data)
    if task_type == 'matching':
        func_key = 'matching_solver'
    else:
        func_key = 'query_solver'
    func = functions_dict.get(func_key)
    out = func(main_model, insight_model, tokenizer, device, data, top_k)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Insight-RAG loop")
    p.add_argument("--domain",        required=True,  help="Target domain")
    p.add_argument("--main_model",    required=True,  help="RAG model name for example gpt-4o")
    p.add_argument("--insight_miner_model_name", required=True,  help="HF model fine-tuned for insight mining")
    p.add_argument("--top_k", type=int, default=1,  help="Number of augmented insights")

    return p.parse_args()

def main() -> None:
    args = parse_args()
    login(HF_TOKEN)


    logger.info(
        "Starting with parameters:\n"
        "  domain        = %s\n"
        "  main_model    = %s\n"
        "  insight_miner_model_name = %s\n",
        args.domain,
        args.main_model,
        args.insight_miner_model_name,
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.insight_miner_model_name)

    # Loading the insight miner model
    inisght_model_name = args.insight_miner_model_name.split('/')[-1]
    path_model = f'output/{args.domain}/{inisght_model_name}'
    insight_model = AutoModelForCausalLM.from_pretrained(args.insight_miner_model_name).to(device)

    tokenizer.pad_token = tokenizer.eos_token
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules = target_modules,
        inference_mode=False,
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.05 
    )

    insight_model = get_peft_model(insight_model, lora_config)
    insight_model = AutoModelForCausalLM.from_pretrained(path_model).to(device)


    for task_type in ["deep", "multi", "matching"]:
        logger.info(f'Starting the {task_type} task!')


        check_path(f'output/tasks/{args.domain}/')
        path_out = f'output/tasks/{args.domain}/{task_type}_{args.main_model}_{args.top_k}.jsonl'
        path_data = f'../data/{args.domain}_{task_type}.jsonl'

        out = insight_rag(args.main_model, insight_model, tokenizer, device, path_data, args.top_k, task_type)
        save_jsonl(path_out, out)

        logger.info(f'Finished the {task_type} task!')



if __name__ == '__main__':
    main()


