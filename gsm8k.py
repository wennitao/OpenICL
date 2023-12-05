from openicl import DatasetReader, PromptTemplate, RandomRetriever, PPLInferencer, GenInferencer
from datasets import load_dataset
from accelerate import Accelerator
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='/home/v-wentaoni/workspace/Llama-2-7b-hf/')
parser.add_argument('--output_path', type=str, default='.')

args = parser.parse_args()

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS

def is_correct(model_completion, gt_example):
    gt_answer = extract_answer(gt_example["answer"])
    assert gt_answer != INVALID_ANS
    return extract_answer(model_completion) == gt_answer

# Define a DatasetReader, loading dataset from huggingface and selecting 5 pieces of data randomly.
# data = DatasetReader('gpt3mix/sst2', input_columns=['text'], output_column='label', ds_size=5)
dataset = load_dataset("gsm8k", "main")
data = DatasetReader (dataset, input_columns=['question'], output_column='answer')

# SST-2 Template Example
# template = PromptTemplate(template={
#                                         0: '</E>Positive Movie Review: </text>',
#                                         1: '</E>Negative Movie Review: </text>' 
#                                    },
#                           column_token_map={'text' : '</text>'},
#                           ice_token='</E>'
#            )

prompt_template = PromptTemplate (template = "You are now asked to solve math problems and here are some examples.\n</E>\nQuestion: </question>\nAnswer: </answer>\nYou can learn to get the answer to this question by looking at the given examples.", 
                                  column_token_map={'question' : '</question>', 'answer' : '</answer>'}, 
                                  ice_token='</E>')
ice_template = PromptTemplate (template = "</E>Question: </question>\nAnswer: </answer>\n", 
                           column_token_map={'question' : '</question>', 'answer' : '</answer>'},
                           ice_token='</E>')

# Accelerate Prepare
# accelerator = Accelerator()

# TopK Retriever
# retriever = TopkRetriever(data, ice_num=2, index_split='train', test_split='test')
retriever = RandomRetriever(data, ice_num=5)

# Define a Inferencer
inferencer = GenInferencer (model_name=args.model_name)
# inferencer = GenInferencer (model_name='/home/v-wentaoni/workspace/llama-2-13b-hf/')
# inferencer = GenInferencer (model_name="google/flan-t5-xxl")

# Inference
# predictions = inferencer.inference(retriever, ice_template=template, output_json_filename='sst2')
predictions, ppls = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template, output_json_filepath=args.output_path, output_json_filename='gsm8k')
print(predictions)

is_correct_list = []
for idx in range (len (predictions)):
    groud_truth = data['test'][retriever.test_idx]['answer']
    # print ("Answer: ", groud_truth)
    # print ("Output: ", predictions[idx])
    ground_truth_answer = extract_answer (groud_truth)
    output_answer = extract_answer (predictions[idx])
    print (ground_truth_answer, output_answer)
    print (ground_truth_answer == output_answer)
    is_correct_list.append (int (ground_truth_answer == output_answer))

import matplotlib.pyplot as plt
import numpy as np
plt.scatter (ppls, is_correct_list)
plt.savefig ("gsm8k_llama7b_k=5_3.png", dpi=300)