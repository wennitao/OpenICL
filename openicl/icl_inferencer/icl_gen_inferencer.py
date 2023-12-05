"""Direct Generation Inferencer"""

import json
import torch
from openicl import PromptTemplate
from openicl.icl_retriever import *
from openicl.icl_evaluator import *
from openicl.icl_inferencer.icl_base_inferencer import BaseInferencer, GenInferencerOutputHandler
from openicl.utils.api_service import *
from openicl.utils.icl_common_utils import get_dataloader, get_generation_prompt_list_from_retriever_indices
from openicl.utils.logging import get_logger
from typing import List, Union, Optional
from tqdm import tqdm
from transformers import PretrainedConfig, AutoConfig
from accelerate import Accelerator

logger = get_logger(__name__)


class GenInferencer(BaseInferencer):
    """Generation In-context Learning Inferencer Class
        In-context Learning Inferencer for Directly Generation.
        
    Attributes:
        model (:obj:`AutoModelForCausalLM`, optional): Local PLM (loaded from Hugging Face), which can be initialized by name or a config class. 
        tokenizer (:obj:`AutoTokenizer` or :obj:`GPT2Tokenizer`, optional): Tokenizer for :obj:`model`.
        max_model_token_num (:obj:`int`, optional): Maximum number of tokenized words allowed by the LM. 
        batch_size (:obj:`int`, optional): Batch size for the :obj:`DataLoader`. 
        accelerator (:obj:`Accelerator`, optional): An instance of the `Accelerator` class, used for multiprocessing.
        output_json_filepath (:obj:`str`, optional): File path for output `JSON` file. 
        output_json_filename (:obj:`str`, optional): File name for output `JSON` file. 
        api_name (:obj:`str`, optional): Name of API service. 
        call_api (:obj:`bool`): If ``True``, an API for LM models will be used, determined by :obj:`api_name`.   
        gen_field_replace_token (:obj:`str`, optional): Used to replace the generation field token when generating prompts.
        generation_kwargs (:obj:`Dict`, optional): Parameters for the :obj:`model.generate()` method. 
    """

    def __init__(self,
                 model_name: Optional[str] = 'gpt2-xl',
                 tokenizer_name: Optional[str] = None,
                 max_model_token_num: Optional[int] = None,
                 model_config: Optional[PretrainedConfig] = None,
                 batch_size: Optional[int] = 1,
                 gen_field_replace_token: Optional[str] = '',
                 generation_kwargs={"max_new_tokens": 100},
                 accelerator: Optional[Accelerator] = None,
                 output_json_filepath: Optional[str] = "./icl_inference_output",
                 output_json_filename: Optional[str] = "predictions",
                 api_name: Optional[str] = None,
                 model_parallel: Optional[bool] = False,
                 **kwargs
                 ) -> None:
        super().__init__(model_name, tokenizer_name, max_model_token_num, model_config, batch_size, accelerator,
                         output_json_filepath, output_json_filename, api_name, model_parallel, **kwargs)
        self.max_position_embeddings = 4096
        self.gen_field_replace_token = gen_field_replace_token
        self.generation_kwargs = generation_kwargs

    def inference(self, retriever: BaseRetriever, ice_template: Optional[PromptTemplate] = None,
                  prompt_template: Optional[PromptTemplate] = None, output_json_filepath: Optional[str] = None,
                  output_json_filename: Optional[str] = None, force_words=None) -> List:
        # 1. Preparation for output logs
        num = len(retriever.test_ds)
        output_handler = GenInferencerOutputHandler(num, self.accelerator)
        index = 0

        if output_json_filepath is None:
            output_json_filepath = self.output_json_filepath
        if output_json_filename is None:
            output_json_filename = self.output_json_filename

        # 2. Get results of retrieval process
        ice_idx_list = retriever.retrieve()
        print (ice_idx_list)

        # 3. Generate prompts for testing input 
        prompt_list = get_generation_prompt_list_from_retriever_indices(ice_idx_list, retriever, self.tokenizer,
                                                                        self.gen_field_replace_token,
                                                                        max_model_token_num=self.max_model_token_num,
                                                                        ice_template=ice_template,
                                                                        prompt_template=prompt_template)
        print (len (prompt_list))
        print (prompt_list[0])
        ppls = []
        for prompt in prompt_list:
            idx = prompt.rfind ("Question")
            text = prompt[:idx]
            question = prompt[idx:]
            ppls.append (self.get_condition_ppl (text, question + "You should answer like this example.\n").cpu().item())
        print (ppls)
        output_handler.save_orgin_prompts(prompt_list)

        # 4. Wrap prompts with Dataloader
        dataloader = get_dataloader(prompt_list, self.batch_size)

        # 5. Inference for prompts in each batch 
        logger.info("Starting inference process...")
        for entry in tqdm(dataloader, disable=not self.is_main_process):
            # 5-1. Inference with local model
            if not self.call_api:
                with torch.no_grad():
                    tokenized_data = self.tokenizer.batch_encode_plus(entry, padding=True, return_tensors='pt').to(
                        self.device)
                    prompt_len = int(tokenized_data.attention_mask.shape[1])
                    if 't5' in self.model_name:
                        prompt_len = 0
                    if force_words is not None:
                        force_words_ids = [
                            self.tokenizer(force_words).input_ids,
                        ]
                        outputs = self.model.generate(input_ids=tokenized_data.input_ids,
                                                      force_words_ids=force_words_ids,
                                                      num_beams=10,
                                                      attention_mask=tokenized_data.attention_mask,
                                                      eos_token_id=self.tokenizer.eos_token_id,
                                                      pad_token_id=self.tokenizer.pad_token_id,
                                                      **self.generation_kwargs)
                    else:
                        outputs = self.model.generate(input_ids=tokenized_data.input_ids,
                                                      attention_mask=tokenized_data.attention_mask,
                                                      eos_token_id=self.tokenizer.eos_token_id,
                                                      pad_token_id=self.tokenizer.pad_token_id,
                                                      **self.generation_kwargs)
                    outputs = outputs.tolist()
                    complete_output = self.tokenizer.batch_decode(outputs[:], skip_special_tokens=True)
                    generated = self.tokenizer.batch_decode([output[prompt_len:] for output in outputs],
                                                            skip_special_tokens=True)
                    print (entry)
                    print (generated)
            # 5-2. Inference with remote API
            else:
                complete_output, generated = api_get_tokens(self.api_name, entry)

            # 5-3. Save current output
            for prediction, output in zip(generated, complete_output):
                output_handler.save_prediction_and_output(prediction, output, index)
                index = index + 1

        # 6. Output 
        output_handler.subprocess_write_to_json(output_json_filepath, output_json_filename)
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        output_handler.merge_to_main_process(output_json_filepath, output_json_filename)
        output_handler.write_to_json(output_json_filepath, output_json_filename)
        return [sample['prediction'] for sample in output_handler.results_dict.values()], ppls

    def get_token_length(self, text: str, add_special_tokens: bool = True):
        return len(
            self.tokenizer(text, add_special_tokens=add_special_tokens).input_ids
        )

    def get_condition_ppl(
        self,
        text: str,
        question: str,
        granularity: str = "sentence",
    ):
        return self.get_ppl(
            text + question,
            granularity=granularity,
            condition_mode="after",
            condition_pos_id=self.get_token_length(text) - 1,
        )

    def get_ppl(
        self,
        text: str,
        granularity: str = "sentence",
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        return_kv=False,
        end=None,
        condition_mode: str = "none",
        condition_pos_id: int = 0,
    ):
        if input_ids is None:
            tokenized_text = self.tokenizer(text, return_tensors="pt")
            input_ids = tokenized_text["input_ids"].to(self.device)
            attention_mask = tokenized_text["attention_mask"].to(self.device)
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
        else:
            past_length = 0
        if end is None:
            end = input_ids.shape[1]
        end = min(end, past_length + self.max_position_embeddings)
        with torch.no_grad():
            response = self.model(
                input_ids[:, past_length:end],
                attention_mask=attention_mask[:, :end],
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = response.past_key_values

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        shift_logits = response.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., past_length + 1 : end].contiguous()
        # Flatten the tokens
        active = (attention_mask[:, past_length:end] == 1)[..., :-1].view(-1)
        active_logits = shift_logits.view(-1, shift_logits.size(-1))[active]
        active_labels = shift_labels.view(-1)[active]
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(active_logits, active_labels)
        if condition_mode == "before":
            loss = loss[:condition_pos_id]
        elif condition_mode == "after":
            loss = loss[condition_pos_id:]
        res = loss.mean() if granularity == "sentence" else loss
        return (res, past_key_values) if return_kv else res

    def __get_ppl(self, input_texts: List[str], mask_length=None):
        if self.call_api:
            return api_get_ppl(self.api_name, input_texts)
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(input_texts, padding=True, return_tensors='pt', truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(
            shift_labels.size())

        if mask_length is not None:
            mask = torch.zeros_like(shift_labels)  # [batch,seqlen]
            for i in range(len(mask)):
                for j in range(mask_length[i] - 1, len(mask[i])):
                    mask[i][j] = 1
            loss = loss * mask

        lens = (inputs["input_ids"] != self.tokenizer.pad_token_id).sum(-1).cpu().numpy()
        if mask_length is not None:
            lens -= np.array(mask_length)
        ce_loss = loss.sum(-1).cpu().detach().numpy() / lens
        return ce_loss
