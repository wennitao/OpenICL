from openicl import DatasetReader, PromptTemplate, RandomRetriever, PPLInferencer, GenInferencer
from datasets import load_dataset
from accelerate import Accelerator
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='/home/v-wentaoni/workspace/Llama-2-7b-hf/')
parser.add_argument('--output_path', type=str, default='.')

args = parser.parse_args()

import string
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()

from collections import Counter
def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

# Define a DatasetReader, loading dataset from huggingface and selecting 5 pieces of data randomly.
# data = DatasetReader('gpt3mix/sst2', input_columns=['text'], output_column='label', ds_size=5)
# dataset = load_dataset("gsm8k", "main")

def prepare_dataset (batch):
    batch['label'] = batch['answer']['value']
    return batch

dataset = load_dataset ("trivia_qa", "rc.nocontext")
dataset = dataset.map (prepare_dataset)
data = DatasetReader (dataset, input_columns=['question'], output_column='label')


# SST-2 Template Example
# template = PromptTemplate(template={
#                                         0: '</E>Positive Movie Review: </text>',
#                                         1: '</E>Negative Movie Review: </text>' 
#                                    },
#                           column_token_map={'text' : '</text>'},
#                           ice_token='</E>'
#            )

prompt_template = PromptTemplate (template = "You are now asked to answer questions and you should give the answer directly. Here is an example.\n</E>\nQuestion: </question>\n", 
                                  column_token_map={'question' : '</question>'}, 
                                  ice_token='</E>')
ice_template = PromptTemplate (template = "</E>Question: </question>\nAnswer: </answer>\n", 
                           column_token_map={'question' : '</question>', 'label' : '</answer>'},
                           ice_token='</E>')

# Accelerate Prepare
# accelerator = Accelerator()

# TopK Retriever
# retriever = TopkRetriever(data, ice_num=2, index_split='train', test_split='test')
retriever = RandomRetriever(data, ice_num=1, index_split='train', test_split='validation')
# retriever.test_idx = 5009

print (data['validation'][retriever.test_idx]['question'])
print (data['validation'][retriever.test_idx]['answer'])

# Define a Inferencer
inferencer = GenInferencer (model_name=args.model_name)
# inferencer = GenInferencer (model_name='/home/v-wentaoni/workspace/llama-2-13b-hf/')
# inferencer = GenInferencer (model_name="google/flan-t5-xxl")

# Inference
# predictions = inferencer.inference(retriever, ice_template=template, output_json_filename='sst2')
predictions, ppls = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template, output_json_filepath=args.output_path, output_json_filename='triviaQA')
# print(predictions)

is_correct_list = []
for idx in range (len (predictions)):
    ground_truth_answer = normalize_answer (data['validation'][retriever.test_idx]['label'])
    aliases = data['validation'][retriever.test_idx]['answer']['normalized_aliases']
    # print ("Prediction: ", predictions[idx])
    # print ("Ground Truth: ", ground_truth_answer)
    predictions[idx] = normalize_answer (predictions[idx])
    find = predictions[idx].find (ground_truth_answer)
    for alias in aliases:
        if predictions[idx].find (alias) != -1:
            find = 1
            break
    is_correct_list.append (1 if find != -1 else 0)

import matplotlib.pyplot as plt
import numpy as np
plt.scatter (ppls, is_correct_list)
plt.savefig ("triviaQA/triviaQA_" + str (retriever.test_idx) + "_llama7b_k=1.png", dpi=300)