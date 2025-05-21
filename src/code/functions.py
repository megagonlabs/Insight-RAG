from prompts import PROMPTS
import numpy as np
import torch
import ast
import openai
import json
import random
import sys
import csv
import os


def clean_insights(text: str) -> str:
    cleaned = (
        text.replace("```", "")
            .replace("json\n", "")
            .replace("json", "")
    )
    cleaned = cleaned.lstrip("\n ").rstrip("\n ")
    return cleaned

def generate(model_name, prompt):
    messages=[{"role": "user", "content": prompt}] 
    response = openai.chat.completions.create(
        model=model_name,
        max_tokens=1000,
        temperature=0.0,
        messages = messages)

    out = response.choices[0].message.content
    return out


def insight_identifier(model_name, task):
    prompt = PROMPTS["insight_identifier"].format(task)
    insights = generate(model_name, prompt)
    return insights 


def insight_miner(pair, model, tokenizer, device, top_k):
    model.eval()
    text, multi = pair["Insight"], pair["Multi-answer"]
    text = text.lower()

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    if multi:
        with torch.no_grad():
            outputs = model.generate(input_ids, max_new_tokens = 120, top_k=top_k, temperature=1.0, do_sample=True, num_return_sequences=top_k, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)

        out = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
        out = '\n'.join(out)

    else:
        with torch.no_grad():
            outputs = model.generate(input_ids, max_new_tokens = 120, top_p=1, do_sample=False, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
        out = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return out


def answering_module(model_name, question, augmented_insights, type_task = 'QA'):
    if type_task == 'QA':
        prompt = PROMPTS["augmented_QA"].format(question, augmented_insights)
    elif type_task == 'matching':
        prompt = PROMPTS["augmented_matching"].format(question, augmented_insights)
    answer = generate(model_name, prompt)
    return answer



def query_solver(model_name, model, tokenizer, device, data, top_k):
    output = []
    for sample in data:
        temp = sample
        question = sample['question']

        task = 'Answer the question: {}'.format(question)


        insights = insight_identifier(model_name, task)
        insights = clean_insights(insights)

        try:
            insights  = json.loads(insights)
        except json.JSONDecodeError as error:
            insights = []

        augmented_insights = ''
        for pair in insights:
            augmented_insights += insight_miner(pair, model, tokenizer, device, top_k) + '\n'


        out = answering_module(model_name, question, augmented_insights, type_task = 'QA')

        temp['Identified_insights'], temp['augmented_insights'], temp['out'] = insights, augmented_insights, out 


        output += [temp]

    return output


def matching_solver(model_name, model, tokenizer, device, data, top_k):
    output = []
    for sample in data:
        temp = sample
        doc_1, doc_2 = sample['doc-1'], sample['doc-2']
        task = '''You are provided with two research papers, Paper-A and Paper-B. Your task is to determine if the papers are relevant enough to be cited by the other.
Paper-A: 
{}

Paper-B:
{}'''.format(doc_1, doc_2)

        task_input = '''Paper-A: 
{}

Paper-B:
{}'''.format(doc_1, doc_2)

        insights = insight_identifier(model_name, task)
        insights = clean_insights(insights)

        try:
            insights  = json.loads(insights)
        except json.JSONDecodeError as error:
            insights = []

        augmented_insights = ''
        for pair in insights:
            augmented_insights += insight_miner(pair, model, tokenizer, device, top_k) + '\n'

        out = answering_module(model_name, task_input, augmented_insights, type_task = 'matching')

        temp['Identified_insights'], temp['augmented_insights'], temp['out'] = insights, augmented_insights, out 


        output += [temp]

    return output
        


functions_dict = {
    'matching_solver': matching_solver,
    'query_solver': query_solver,
}
