from abc import ABC, abstractmethod
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import (AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification,
                          AutoTokenizer, LlamaModel, LlamaPreTrainedModel, LlamaTokenizer,
                          PreTrainedTokenizerFast, pipeline)
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from cheems.train.dataset import build_sample
from cheems.train.model import get_reward_model_class
from cheems.train.tokenizer import get_tokenizer


class RMPredictor(ABC):
    @abstractmethod
    def reward_fn(self, conv) -> float:
        pass


class DefaultRMPredictor(RMPredictor):
    def __init__(self, model_path):
        self.tokenizer = get_tokenizer(model_path)
        RewardModelClass = get_reward_model_class(model_path)
        self.model = RewardModelClass.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,  # 使用flash_attention必须要bf16
            device_map="balanced",
        )

    def reward_fn(self, conv):
        output_dict = build_sample(self.tokenizer, input_or_conv=conv, tokenize=True)
        with torch.no_grad():
            return self.model.inference(
                input_ids=torch.as_tensor([output_dict['input_ids']]).to(self.model.device),
                attention_mask=torch.as_tensor([output_dict['attention_mask']]).to(self.model.device),
            )[0].item()


class SkyworkRMPredictor(RMPredictor):
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
            num_labels=1,
        )

    def reward_fn(self, conv):
        conv = self.tokenizer.apply_chat_template(conv, tokenize=False)
        tokenized = self.tokenizer(conv, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            return self.model(**tokenized).logits[0][0].item()


class QRMPredictor(RMPredictor):
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True
        )

    def reward_fn(self, conv):
        conv = self.tokenizer.apply_chat_template(conv, tokenize=False)
        tokenized = self.tokenizer(conv, return_tensors="pt").to("cuda")
        with torch.no_grad():
            return self.model(**tokenized).score.item()


class NemotronPredictor(RMPredictor):
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def reward_fn(self, conv):
        conv = self.tokenizer.apply_chat_template(conv, tokenize=False)
        tokenized = self.tokenizer(conv, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            response_token_ids = self.model.generate(
                tokenized['input_ids'],
                attention_mask=tokenized['attention_mask'],
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
            )
            return response_token_ids['scores'][0][0][0].item()


class URMPredictor(RMPredictor):
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            device_map='auto',
            trust_remote_code=True,
        )

    def reward_fn(self, conv):
        conv = self.tokenizer.apply_chat_template(conv, tokenize=False)
        tokenized = self.tokenizer(conv, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            returned = self.model(tokenized['input_ids'], attention_mask=tokenized['attention_mask'])
            return returned.logits[0][0].item()


class GRMPredictor(RMPredictor):
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained('Ray2333/GRM-Llama3-8B-rewardmodel-ft')
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map='auto',
        )

    def reward_fn(self, conv):
        conv_template = self.tokenizer.apply_chat_template(conv, tokenize=False)
        kwargs = {"padding": 'max_length', "truncation": True, "return_tensors": "pt"}
        tokens = self.tokenizer.encode_plus(conv_template, **kwargs)
        with torch.no_grad():
            reward_tensor = self.model(
                tokens["input_ids"][0].view(1, -1).to(self.model.device),
                attention_mask=tokens["attention_mask"][0].view(1, -1).to(self.model.device)
            )[0]
            return reward_tensor.cpu().detach().item()


class ArmoRMPredictor(RMPredictor):
    def __init__(self, model_path):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True,
        )
        self.truncation = True
        self.device = self.model.device
        self.max_length = 4096

    def reward_fn(self, conv):
        input_ids = self.tokenizer.apply_chat_template(
            conv,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
            return output.score.float().item()


class InternLM2Predictor(RMPredictor):
    def __init__(self, model_path):
        self.model = AutoModel.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def reward_fn(self, conv):
        with torch.no_grad():
            return self.model.get_score(self.tokenizer, conv)


class Ray2333Predictor(RMPredictor):
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=1,
            torch_dtype=torch.float16,
            device_map=0,
        )

    def reward_fn(self, conv):
        conv_template = self.tokenizer.apply_chat_template(conv, tokenize=False)
        kwargs = {"padding": 'max_length', "truncation": True, "return_tensors": "pt"}
        tokens = self.tokenizer.encode_plus(conv_template, **kwargs)
        with torch.no_grad():
            reward_tensor = self.model(
                tokens["input_ids"][0].to(self.model.device),
                attention_mask=tokens["attention_mask"][0].to(self.model.device)
            ).logits.reshape(-1)
            return reward_tensor.cpu().detach().item()


class BTRMPredictor(RMPredictor):
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pipe = pipeline(
            "sentiment-analysis",
            model=model_path,  # 模型路径
            device_map="auto",
            tokenizer=self.tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16}
        )

    def reward_fn(self, conv):
        test_text = self.tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=False
        )
        pipe_output = self.pipe(test_text, **{
            "top_k": None,
            "function_to_apply": "none",
            "batch_size": 1
        })
        return pipe_output[0]["score"]


class SentimentPredictor(RMPredictor):
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pipe = pipeline(
            "sentiment-analysis",
            model=model_path,
            device_map="auto",
            tokenizer=self.tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16}
        )

    def reward_fn(self, conv):
        test_texts = [
            self.tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False
            ).replace(self.tokenizer.bos_token, "")
        ]
        pipe_outputs = self.pipe(test_texts, **{
            "return_all_scores": True,
            "function_to_apply": "none",
            "batch_size": 1
        })
        rewards = [output[0]["score"] for output in pipe_outputs]
        return rewards[0]


class INFORMPredictor(RMPredictor):
    class INFORMForSequenceClassification(LlamaPreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.num_labels = config.num_labels
            self.model = LlamaModel(config)
            self.score = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, self.num_labels)
            )
            # Initialize weights and apply final processing
            self.post_init()

        def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ):

            transformer_outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
            )
            hidden_states = transformer_outputs[0]
            logits = self.score(hidden_states)

            if input_ids is not None:
                batch_size = input_ids.shape[0]
            else:
                batch_size = inputs_embeds.shape[0]

            if self.config.pad_token_id is None and batch_size != 1:
                raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
            if self.config.pad_token_id is None:
                sequence_lengths = -1
            else:
                if input_ids is not None:
                    # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                    sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                    sequence_lengths = sequence_lengths % input_ids.shape[-1]
                    sequence_lengths = sequence_lengths.to(logits.device)
                else:
                    sequence_lengths = -1

            pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

            loss = None
            return SequenceClassifierOutputWithPast(
                loss=loss,
                logits=pooled_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
            )

    def __init__(self, model_name):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
        self.orm = self.INFORMForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
            num_labels=1,
        )

    def reward_fn(self, conv):
        conv_tokenized = self.tokenizer.apply_chat_template(conv, tokenize=True, return_tensors="pt").to("cuda")
        return self.orm(conv_tokenized).logits[0][0].item()


class DebertaV3Predictor(RMPredictor):
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, device_map="cuda")

    def reward_fn(self, conv):
        question, answer = conv[0]['content'], conv[1]['content']
        inputs = self.tokenizer(question, answer, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            score = self.model(**inputs).logits[0].cpu().detach().item()
        return score


class ZiyaPredictor(RMPredictor):
    def __init__(self, model_path):
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path, add_eos_token=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, trust_remote_code=True
        ).eval().half().cuda()

    def reward_fn(self, conv):
        prefix_user = "Human:"
        prefix_bot = "\n\nAssistant:"
        text = prefix_user + conv[0]['content'] + prefix_bot + conv[1]['content']
        batch = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        with torch.no_grad():
            reward = self.model(batch['input_ids'].cuda(), attention_mask=batch['attention_mask'].cuda()).item()
        return reward


PREDICTOR_MAP = {
    "infly/INF-ORM-Llama3.1-70B": INFORMPredictor,
    "Skywork/Skywork-Reward-Gemma-2-27B-v0.2": SkyworkRMPredictor,
    "Skywork/Skywork-Reward-Gemma-2-27B": SkyworkRMPredictor,
    "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2": SkyworkRMPredictor,
    "Skywork/Skywork-Reward-Llama-3.1-8B": SkyworkRMPredictor,
    "nicolinho/QRM-Llama3.1-8B": QRMPredictor,
    "LxzGordon/URM-LLaMa-3.1-8B": URMPredictor,
    "LxzGordon/URM-LLaMa-3-8B": URMPredictor,
    "Ray2333/GRM-gemma2-2B-rewardmodel-ft": GRMPredictor,
    "Ray2333/GRM-Llama3-8B-rewardmodel-ft": GRMPredictor,
    "Ray2333/GRM-llama3-8B-distill": GRMPredictor,
    "Ray2333/GRM-Gemma-2B-rewardmodel-ft": GRMPredictor,
    "Ray2333/Gemma-2B-rewardmodel-ft": GRMPredictor,
    "RLHFlow/ArmoRM-Llama3-8B-v0.1": ArmoRMPredictor,
    "internlm/internlm2-20b-reward": InternLM2Predictor,
    "internlm/internlm2-1_8b-reward": InternLM2Predictor,
    "internlm/internlm2-7b-reward": InternLM2Predictor,
    "CIR-AMS/BTRM_Qwen2_7b_0613": BTRMPredictor,
    "NCSOFT/Llama-3-OffsetBias-RM-8B": SentimentPredictor,
    "sfairXC/FsfairX-LLaMA3-RM-v0.1": SentimentPredictor,
    "weqweasdas/RM-Gemma-2B": SentimentPredictor,
    "weqweasdas/RM-Gemma-7B": SentimentPredictor,
    "weqweasdas/RM-Mistral-7B": SentimentPredictor,
    "OpenAssistant/reward-model-deberta-v3-large-v2": DebertaV3Predictor,
    "IDEA-CCNL/Ziya-LLaMA-7B-Reward": ZiyaPredictor,
    "nvidia/Llama-3.1-Nemotron-70B-Reward-HF": NemotronPredictor,
}
