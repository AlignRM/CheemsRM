import copy
import json
from abc import ABC, abstractmethod
from typing import List, Union

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

PROMPT_TEMPLATE = """"Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user\'s instructions and answers the user\'s question better.
Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible.
Please directly output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better.

[User Question]
{question}

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
"""


class GenerativePredictor(ABC):
    @abstractmethod
    def compare_fn(
        self,
        question: Union[str, List[str]],
        answer1: Union[str, List[str]],
        answer2: Union[str, List[str]],
    ) -> Union[str, List[str]]:
        pass


class VllmPredictor(GenerativePredictor):
    def __init__(self, model_name, tensor_parallel_size: int = 1, max_model_len: int = 8192):
        self.llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size, max_model_len=max_model_len)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sampling_params = SamplingParams(temperature=0.)

    def compare_fn(self, question: str, answer_a: str, answer_b: str) -> Union[str, List[str]]:
        if isinstance(question, str):
            question, answer_a, answer_b = [question], [answer_a], [answer_b]  # type: ignore

        judge_prompts = []
        for ques, ans_a, ans_b in zip(question, answer_a, answer_b):
            judge_prompt = PROMPT_TEMPLATE.format(question=ques, answer_a=ans_a, answer_b=ans_b)
            judge_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": judge_prompt}],
                tokenize=False, add_generation_prompt=True,
            )
            judge_prompts.append(judge_prompt)
            # print(judge_prompt)

        outputs = self.llm.generate(judge_prompts, self.sampling_params, use_tqdm=False)
        judgments = [output.outputs[0].text for output in outputs]

        results = []
        for judgment in judgments:
            if "[[A]]" in judgment:
                results.append("A")
            elif "[[B]]" in judgment:
                results.append("B")
            else:
                results.append("error")
        return results[0] if len(results) == 1 else results


class SelfTaughtPredictor(VllmPredictor):
    SELF_TAUGHT_WITH_SYSTEM_PROMPT = [
        {
            "role": "system",
            "content": 'Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user\'s instructions and answers the user\'s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \\"[[A]]\\" if assistant A is better, \\"[[B]]\\" if assistant B is better.',
        },
        {
            "role": "user",
            "content": """[User Question]
{input}

[The Start of Assistant A's Answer]
{response_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{response_b}
[The End of Assistant B's Answer]
""",
        },
    ]

    def compare_fn(self, question: str, answer_a: str, answer_b: str) -> Union[str, List[str]]:
        if isinstance(question, str):
            question, answer_a, answer_b = [question], [answer_a], [answer_b]  # type: ignore

        judge_prompts = []
        for ques, ans_a, ans_b in zip(question, answer_a, answer_b):
            conversation = copy.copy(self.SELF_TAUGHT_WITH_SYSTEM_PROMPT)
            conversation[-1]["content"] = conversation[-1]["content"].format(input=ques, response_a=ans_a, response_b=ans_b)
            judge_prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            judge_prompts.append(judge_prompt)
            # print(judge_prompt)

        outputs = self.llm.generate(judge_prompts, self.sampling_params, use_tqdm=False)
        judgments = [output.outputs[0].text for output in outputs]

        results = []
        for judgment in judgments:
            if "[[A]]" in judgment:
                results.append("A")
            elif "[[B]]" in judgment:
                results.append("B")
            else:
                results.append("error")
        return results[0] if len(results) == 1 else results


class ConQwen2Predictor(VllmPredictor):
    CON_J_PROMPT = """作为一个评价专家，给定一个问题和它的两个可能的回答，请选出哪一个回答在连贯性、准确性、覆盖度和上述定义的整体质量方面最为符合。请用JSON格式输出你的判断, 其中"原因"是你提供的解释，"更好的回答"是整数类型的1或2，例如{{"原因": "你的解释", "更好的回答": 1}}。以下是问题和候选回答的内容：
    \n问题：{instruction}
回答1：{output_1}
回答2：{output_2}"""

    def compare_fn(self, question: str, answer_a: str, answer_b: str) -> Union[str, List[str]]:
        if isinstance(question, str):
            question, answer_a, answer_b = [question], [answer_a], [answer_b]  # type: ignore

        judge_prompts = []
        for ques, ans_a, ans_b in zip(question, answer_a, answer_b):
            user_prompt = self.CON_J_PROMPT.format(instruction=ques, output_1=ans_a, output_2=ans_b)
            system_prompt = ""
            messages = [
                {"role": "system", "content": system_prompt, },
                {"role": "user", "content": user_prompt},
            ]
            judge_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            judge_prompts.append(judge_prompt)
            # print(judge_prompt)

        outputs = self.llm.generate(judge_prompts, self.sampling_params, use_tqdm=False)
        judgments = [output.outputs[0].text for output in outputs]

        results = []
        for judgment in judgments:
            try:
                result = json.loads(judgment.strip())["更好的回答"]
                if int(result) == 1:
                    results.append("A")
                elif int(result) == 2:
                    results.append("B")
                else:
                    results.append("error")
            except Exception:
                results.append("error")
        return results[0] if len(results) == 1 else results


class OffsetBias2Predictor(VllmPredictor):
    prompt_template = """You are a helpful assistant in evaluating the quality of the outputs for a given instruction. Your goal is to select the best output for the given instruction.

Select the Output (a) or Output (b) that is better for the given instruction. The two outputs are generated by two different AI chatbots respectively.
Do NOT provide any explanation for your choice.
Do NOT say both / neither are good.
You should answer using ONLY “Output (a)” or “Output (b)”. Do NOT output any other words.
Here are some rules of the evaluation:
(1) You should prioritize evaluating whether the output honestly/precisely/closely executes the instruction, then consider its helpfulness, accuracy, level of detail, harmlessness, etc.
(2) Outputs should NOT contain more/less than what the instruction asks for, as such outputs do NOT precisely execute the instruction.
(3) You should avoid any potential bias and your judgment should be as objective as possible. For example, the order in which the outputs were presented should NOT affect your judgment, as Output (a) and Output (b) are **equally likely** to be the better.

# Instruction:
{input}
# Output (a):
{output_1}
# Output (b):
{output_2}
# Which is better, Output (a) or Output (b)? Your response should be either “Output (a)” or “Output (b)”:"""

    def compare_fn(self, question: str, answer_a: str, answer_b: str) -> Union[str, List[str]]:
        if isinstance(question, str):
            question, answer_a, answer_b = [question], [answer_a], [answer_b]  # type: ignore

        judge_prompts = []
        for ques, ans_a, ans_b in zip(question, answer_a, answer_b):
            user_message = self.prompt_template.format(input=ques, output_1=ans_a, output_2=ans_b)
            conversation = [{"role": "user", "content": user_message}]
            judge_prompt = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            judge_prompts.append(judge_prompt)
            # print(judge_prompt)

        outputs = self.llm.generate(judge_prompts, self.sampling_params, use_tqdm=False)
        judgments = [output.outputs[0].text for output in outputs]

        results = []
        for judgment in judgments:
            if judgment == "Output (a)":
                results.append("A")
            elif judgment == "Output (b)":
                results.append("B")
            else:
                results.append("error")
        return results[0] if len(results) == 1 else results


class CompassJudgerPredictor(VllmPredictor):
    prompt_template = """**Input**: ```Please read the dialogue between the two assistants and the user to determine which assistant performed better during the conversation.Here is the dialogue content:
[Dialogue Begin]
User: {input}
Assistant A: {output_1}
Assistant B: {output_2}
[Dialogue End]
If you believe Assistant A performed better, please output A directly.\nIf you believe Assistant B performed better, please output B directly.\nDo not output any other content, just the option. Please output:```"""

    def compare_fn(self, question: str, answer_a: str, answer_b: str) -> Union[str, List[str]]:
        if isinstance(question, str):
            question, answer_a, answer_b = [question], [answer_a], [answer_b]  # type: ignore

        judge_prompts = []
        for ques, ans_a, ans_b in zip(question, answer_a, answer_b):
            user_message = self.prompt_template.format(input=ques, output_1=ans_a, output_2=ans_b)
            conversation = [{"role": "user", "content": user_message}]
            judge_prompt = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            judge_prompts.append(judge_prompt)
            # print(judge_prompt)

        outputs = self.llm.generate(judge_prompts, self.sampling_params, use_tqdm=False)
        judgments = [output.outputs[0].text for output in outputs]

        results = []
        for judgment in judgments:
            if judgment.strip().lower().endswith("a"):
                results.append("A")
            elif judgment.strip().lower().endswith("b"):
                results.append("B")
            else:
                results.append("error")
        return results[0] if len(results) == 1 else results


PREDICTOR_MAP = {
    "Qwen/Qwen2.5-0.5B-Instruct": VllmPredictor,
    "Qwen/Qwen2.5-1.5B-Instruct": VllmPredictor,
    "Qwen/Qwen2.5-3B-Instruct": VllmPredictor,
    "Qwen/Qwen2.5-7B-Instruct": VllmPredictor,
    "Qwen/Qwen2.5-14B-Instruct": VllmPredictor,
    "Qwen/Qwen2.5-32B-Instruct": VllmPredictor,
    "Qwen/Qwen2.5-72B-Instruct": VllmPredictor,
    "meta-llama/Llama-3.1-8B-Instruct": VllmPredictor,
    "meta-llama/Llama-3.1-70B-Instruct": VllmPredictor,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": VllmPredictor,
    "Skywork/Skywork-Critic-Llama-3.1-8B": VllmPredictor,
    "Skywork/Skywork-Critic-Llama-3.1-70B": VllmPredictor,
    "facebook/Self-taught-evaluator-llama3.1-70B": SelfTaughtPredictor,
    "ZiyiYe/Con-J-Qwen2-7B": ConQwen2Predictor,
    "NCSOFT/Llama-3-OffsetBias-8B": OffsetBias2Predictor,
    "opencompass/CompassJudger-1-1.5B-Instruct": CompassJudgerPredictor,
    "opencompass/CompassJudger-1-14B-Instruct": CompassJudgerPredictor,
    "opencompass/CompassJudger-1-32B-Instruct": CompassJudgerPredictor,
}
