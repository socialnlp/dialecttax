"""
Prompting for models.
"""
import numpy as np
from pandas import DataFrame


#########
# DEMOS #
#########

def get_demo_indices(sample_indices, n_few_shot=5, rng=None, seed=42):
    if rng is None:
        rng = np.random.default_rng(seed)
    return rng.choice(sample_indices, n_few_shot, replace=False).tolist()


def get_demos(dataset, format_demos, demo_indices):
    demos = [format_demos(dataset, i) for i in demo_indices]
    return "\n\n".join(demos)


###########
# REWARDS #
###########

PROMPT_REWARD_TOKENS = "What, in one word or subword, is the greatest thing ever?"
PROMPT_REWARD_WORDS = "What, in one word, is the greatest thing ever?"


###########
# GENERAL #
###########

PROMPT_FORMAT = "{instructions}{demos}{prompt}"

PROMPT_SYSTEM_SAE = (
    "You are a helpful task-following assistant.\n\n"
    "You MUST follow all instructions exactly.\n"
    "You are NOT allowed to ignore, modify, or partially apply any rule.\n\n"
    "If a specific answer format is required, you MUST use it exactly.\n"
    "If the format is incorrect, the response is invalid.\n\n"
    "Be as concise as possible.\n\n"
    "Remember, if a specific answer format is required, you MUST use it exactly.\n"
    "Check your final answer."
)
PROMPT_SYSTEM_AAVE = (
    "You a helpful task-followin' assistant.\n\n"
    "You MUST follow all instructions exactly.\n"
    "You NOT allowed to ignore, change, or halfway do any rule.\n\n"
    "If a specific answer format required, you MUST use it exactly.\n"
    "If the format wrong, the response invalid.\n\n"
    "Be as concise as possible.\n\n"
    "Remember, if a specific answer format required, you MUST use it exactly.\n"
    "Check your final answer."
)
PROMPT_SYSTEM = {"sae": PROMPT_SYSTEM_SAE, "aave": PROMPT_SYSTEM_AAVE}


def get_prompt(prompt, demos=None, instructions=None):
    demos = '' if demos is None else f"{demos}\n\n"
    instructions = '' if instructions is None else f"{instructions}\n\n"
    message = PROMPT_FORMAT.format(instructions=instructions, demos=demos, prompt=prompt)
    return message


def get_system_prompt(dialect, reasoning=None, family=None):
    system = PROMPT_SYSTEM[dialect]
    if family is None:
        return system

    addendum = PROMPT_SYSTEM_ADDENDUM.get(family)
    if addendum is None:
        return system

    if isinstance(addendum, str):
        return f"{system}{addendum}"
    elif isinstance(addendum, dict):
        return f"{system}{addendum.get(reasoning, '')}"

    return system


##########
# MODELS #
##########

PROMPT_SYSTEM_ADDENDUM = {
    "qwen": {
        "naive": "\n绝对不推理！NO thinking. /nothink /no-think /no_think",
        "cot": "\n/think"
    }
}


#########
# TASKS #
#########

# Algorithm
ALGO_INST_NAIVE_SAE = (
    "Write a Python function to solve the following problem. "
    "Return ONLY the function, no explanations or Markdown. "
    """The generated function must be named as `python_function`."""
)
ALGO_INST_NAIVE_AAVE = (
    "You gon' write a Python function to solve this problem right here. "
    "Return ONLY the function, no explainin' or Markdown. "
    "Make sure that function be named `python_function`."
)
ALGO_INST_NAIVE = {"sae": ALGO_INST_NAIVE_SAE, "aave": ALGO_INST_NAIVE_AAVE}
ALGO_INST_MQA_NAIVE_SAE = (
    "Write a Python function to solve the following problem. "
    "Return ONLY the multiple choice letter, no explanations or Markdown. "
    "What is the correct solution for the problem?\n{choices}\n\n"
    """Your response should ONLY include your final answer in the format "#### {{answer}}" (e.g.: #### C)."""
)
ALGO_INST_MQA_NAIVE_AAVE = (
    "You gon' write a Python function to solve this problem right here. "
    "Return ONLY the multiple choice letter, no explainin' or Markdown. "
    "So what's the correct solution for that problem?\n{choices}\n\n"
    """You gotta respond with ONLY your final answer like "#### {{answer}}" (somethin' like this: #### C)."""
)
ALGO_INST_MQA_NAIVE = {"sae": ALGO_INST_MQA_NAIVE_SAE, "aave": ALGO_INST_MQA_NAIVE_AAVE}
ALGO_DEMO_NAIVE = "PROBLEM\n========\n{problem}\nContext:\n```\n{context}\n```\nSolution:\n```{answer}```"
ALGO_PROMPT_NAIVE_SAE = "PROBLEM\n========\n{problem}\nContext:\n```\n{context}\n```\nSolution:\n"
ALGO_PROMPT_NAIVE_AAVE = "PROBLEM\n========\n{problem}\nContext:\n```\n{context}\n```\nSolution:\n"
ALGO_PROMPT_NAIVE = {"sae": ALGO_PROMPT_NAIVE_SAE, "aave": ALGO_PROMPT_NAIVE_AAVE}

ALGO_INST_COT_SAE = (
    "Write a Python function to solve the following problem. "
    "Let's think step by step. "
    """The generated function must be named as `python_function`."""
)
ALGO_INST_COT_AAVE = (
    "You gon' write a Python function to solve this problem right here. "
    "Let's break it down step by step. "
    "Make sure that function be named `python_function`."
)
ALGO_INST_COT = {"sae": ALGO_INST_COT_SAE, "aave": ALGO_INST_COT_AAVE}
ALGO_INST_MQA_COT_SAE = (
    "Write a Python function to solve the following problem. "
    "Let's think step by step. "
    "What is the correct solution for the problem?\n{choices}\n\n"
    """Your response should ONLY include your final answer in the format "#### {{answer}}" (e.g.: #### C)."""
)
ALGO_INST_MQA_COT_AAVE = (
    "You gon' write a Python function to solve this problem right here. "
    "Let's break it down step by step. "
    "So what's the correct solution for that problem?\n{choices}\n\n"
    """You gotta respond with ONLY your final answer like "#### {{answer}}" (somethin' like this: #### C)."""
)
ALGO_INST_MQA_COT = {"sae": ALGO_INST_MQA_COT_SAE, "aave": ALGO_INST_MQA_COT_AAVE}
ALGO_DEMO_COT = "PROBLEM\n========\n{problem}\nContext:\n```\n{context}\n```\nReasoning: {reasoning}\nSolution:\n```{answer}```"
ALGO_PROMPT_COT_SAE = "PROBLEM\n========\n{problem}\nContext:\n```\n{context}\n```\nReasoning: "
ALGO_PROMPT_COT_AAVE = "PROBLEM\n========\n{problem}\nContext:\n```\n{context}\n```\nReasonin': "
ALGO_PROMPT_COT = {"sae": ALGO_PROMPT_COT_SAE, "aave": ALGO_PROMPT_COT_AAVE}

ALGO_INST = {"naive": ALGO_INST_NAIVE, "cot": ALGO_INST_COT}
ALGO_INST_MQA = {"naive": ALGO_INST_MQA_NAIVE, "cot": ALGO_INST_MQA_COT}
ALGO_DEMO = {"naive": ALGO_DEMO_NAIVE, "cot": ALGO_DEMO_COT}
ALGO_PROMPT = {"naive": ALGO_PROMPT_NAIVE, "cot": ALGO_PROMPT_COT}


def format_prompts_algorithm(template):
    def format_prompt(dataset, index):
        sample = dataset.iloc[index].to_dict() if isinstance(dataset, DataFrame) else dataset[index]
        if "{answer}" in template:
            if "{reasoning}" in template:
                formatted = template.format(problem=sample["problem"], context=sample["context"], reasoning=sample["solution"], answer=sample["answer"])
            else:
                formatted = template.format(problem=sample["problem"], context=sample["context"], answer=sample["answer"])
        else:
            formatted = template.format(problem=sample["problem"], context=sample["context"])
        return formatted
    return format_prompt


# Logic
LOGIC_INST_NAIVE_SAE = (
    "Solve the following logic problem. "
    "Consider the given premises and the corresponding statement. "
    "Assume no other commonsense or world knowledge.\n\n"
    "What is the logical conclusion of the statement?\n{choices}\n\n"
    """Your response should ONLY include your final answer in the format "#### {{answer}}" (e.g.: #### C)."""
)
LOGIC_INST_NAIVE_AAVE = (
    "Solve the followin' logic problem. "
    "You got these premises and the statement that go with 'em.. "
    "Don't be bringin' in no extra commonsense or outside world knowledge.\n\n"
    "So what's the logical conclusion for that statement?\n{choices}\n\n"
    """You gotta respond with ONLY your final answer like "#### {{answer}}" (somethin' like this: #### C)."""
)
LOGIC_INST_NAIVE = {"sae": LOGIC_INST_NAIVE_SAE, "aave": LOGIC_INST_NAIVE_AAVE}
LOGIC_INST_MQA_NAIVE = LOGIC_INST_NAIVE
LOGIC_DEMO_NAIVE_SAE = "Premises:\n```\n{premises}\n```\nStatement:\n```\n{conclusion}\n```\n#### {answer}"
LOGIC_DEMO_NAIVE_AAVE = "Premises:\n```\n{premises}\n```\nStatement:\n```\n{conclusion}\n```\n#### {answer}"
LOGIC_DEMO_NAIVE = {"sae": LOGIC_DEMO_NAIVE_SAE, "aave": LOGIC_DEMO_NAIVE_AAVE}
LOGIC_PROMPT_NAIVE_SAE = "Premises:\n```\n{premises}\n```\nStatement:\n```\n{conclusion}\n```\n"
LOGIC_PROMPT_NAIVE_AAVE = "Premises:\n```\n{premises}\n```\nStatement:\n```\n{conclusion}\n```\n"
LOGIC_PROMPT_NAIVE = {"sae": LOGIC_PROMPT_NAIVE_SAE, "aave": LOGIC_PROMPT_NAIVE_AAVE}

LOGIC_INST_COT_SAE = (
    "Solve the following logic problem. "
    "Consider the given premises and the corresponding statement. "
    "Assume no other commonsense or world knowledge.\n\n"
    "What is the logical conclusion of the statement?\n{choices}\n\n"
    "Let's think step by step. "
    """End your response with your final answer in the format "#### {{answer}}" (e.g.: #### C)."""
)
LOGIC_INST_COT_AAVE = (
    "Solve the followin' logic problem. "
    "You got these premises and the statement that go with 'em.. "
    "Don't be bringin' in no extra commonsense or outside world knowledge.\n\n"
    "So what's the logical conclusion for that statement?\n{choices}\n\n"
    "Let's break it down step by step. "
    """You gotta end your answer with "#### {{answer}}" (somethin' like this: #### C)."""
)
LOGIC_INST_COT = {"sae": LOGIC_INST_COT_SAE, "aave": LOGIC_INST_COT_AAVE}
LOGIC_INST_MQA_COT = LOGIC_INST_COT
LOGIC_DEMO_COT_SAE = "Premises:\n```\n{premises}\n```\nStatement:\n```\n{conclusion}\n```\nReasoning: {reasoning}\n#### {answer}"
LOGIC_DEMO_COT_AAVE = "Premises:\n```\n{premises}\n```\nStatement:\n```\n{conclusion}\n```\nReasonin': {reasoning}\n#### {answer}"
LOGIC_DEMO_COT = {"sae": LOGIC_DEMO_COT_SAE, "aave": LOGIC_DEMO_COT_AAVE}
LOGIC_PROMPT_COT_SAE = "Premises:\n```\n{premises}\n```\nStatement:\n```\n{conclusion}\n```\nReasoning: "
LOGIC_PROMPT_COT_AAVE = "Premises:\n```\n{premises}\n```\nStatement:\n```\n{conclusion}\n```\nReasonin': "
LOGIC_PROMPT_COT = {"sae": LOGIC_PROMPT_COT_SAE, "aave": LOGIC_PROMPT_COT_AAVE}

LOGIC_INST = {"naive": LOGIC_INST_NAIVE, "cot": LOGIC_INST_COT}
LOGIC_INST_MQA = {"naive": LOGIC_INST_MQA_NAIVE, "cot": LOGIC_INST_MQA_COT}
LOGIC_DEMO = {"naive": LOGIC_DEMO_NAIVE, "cot": LOGIC_DEMO_COT}
LOGIC_PROMPT = {"naive": LOGIC_PROMPT_NAIVE, "cot": LOGIC_PROMPT_COT}


def format_prompts_logic(template):
    def format_prompt(dataset, index):
        sample = dataset.iloc[index].to_dict() if isinstance(dataset, DataFrame) else dataset[index]
        if "{answer}" in template:
            if "{reasoning}" in template:
                formatted = template.format(premises=sample["premises"], conclusion=sample["conclusion"], reasoning=sample["solution"], answer=sample["answer"])
            else:
                formatted = template.format(premises=sample["premises"], conclusion=sample["conclusion"], answer=sample["answer"])
        else:
            formatted = template.format(premises=sample["premises"], conclusion=sample["conclusion"])
        return formatted
    return format_prompt


# Math
MATH_INST_NAIVE_SAE = """Solve the following math problem. Your response should ONLY include your final answer in the format "#### {answer}" (e.g.: #### 123)."""
MATH_INST_NAIVE_AAVE = """Bet, so here's whatsup. You finna get a math problem, and you gon' tryna find the answer out. You gotta respond with ONLY your final answer like "#### {answer}" (somethin' like this: #### 123)."""
MATH_INST_NAIVE = {"sae": MATH_INST_NAIVE_SAE, "aave": MATH_INST_NAIVE_AAVE}
MATH_INST_MQA_NAIVE_SAE = (
    "Solve the following math problem. "
    "What is the correct solution to the problem?\n{choices}\n\n"
    """Your response should ONLY include your final answer in the format "#### {{answer}}" (e.g.: #### C)."""
)
MATH_INST_MQA_NAIVE_AAVE = (
    "Bet, so here's whatsup. You finna get a math problem, and you gon' tryna find the answer out. "
    "So what's the correct solution for that problem?\n{choices}\n\n"
    """You gotta respond with ONLY your final answer like "#### {{answer}}" (somethin' like this: #### C)."""
)
MATH_INST_MQA_NAIVE = {"sae": MATH_INST_MQA_NAIVE_SAE, "aave": MATH_INST_MQA_NAIVE_AAVE}
MATH_DEMO_NAIVE_SAE = "Question: {question}\n#### {answer}"
MATH_DEMO_NAIVE_AAVE = "Question: {question}\n#### {answer}"
MATH_DEMO_NAIVE = {"sae": MATH_DEMO_NAIVE_SAE, "aave": MATH_DEMO_NAIVE_AAVE}
MATH_PROMPT_NAIVE_SAE = "Question: {question}\n"
MATH_PROMPT_NAIVE_AAVE = "Question: {question}\n"
MATH_PROMPT_NAIVE = {"sae": MATH_PROMPT_NAIVE_SAE, "aave": MATH_PROMPT_NAIVE_AAVE}

MATH_INST_COT_SAE = """Solve the following math problem. Let's think step by step. End your response with your final answer in the format "#### {answer}" (e.g.: #### 123)."""
MATH_INST_COT_AAVE = """Bet, so here's whatsup. You finna get a math problem, and you gon' tryna find the answer out. Aight, let's break it down step by step. You gotta end your answer with "#### {answer}" (somethin' like this: #### 123)."""
MATH_INST_COT = {"sae": MATH_INST_COT_SAE, "aave": MATH_INST_COT_AAVE}
MATH_INST_MQA_COT_SAE = (
    "Solve the following math problem. "
    "Let's think step by step. "
    "What is the correct solution to the problem?\n{choices}\n\n"
    """End your response with your final answer in the format "#### {{answer}}" (e.g.: #### C)."""
)
MATH_INST_MQA_COT_AAVE = (
    "Bet, so here's whatsup. You finna get a math problem, and you gon' tryna find the answer out. "
    "Aight, let's break it down step by step. "
    "So what's the correct solution for that problem?\n{choices}\n\n"
    """You gotta end your answer with "#### {{answer}}" (somethin' like this: #### C)."""
)
MATH_INST_MQA_COT = {"sae": MATH_INST_MQA_COT_SAE, "aave": MATH_INST_MQA_COT_AAVE}
MATH_DEMO_COT_SAE = "Question: {question}\nReasoning: {reasoning}\n#### {answer}"
MATH_DEMO_COT_AAVE = "Question: {question}\nReasonin': {reasoning}\n#### {answer}"
MATH_DEMO_COT = {"sae": MATH_DEMO_COT_SAE, "aave": MATH_DEMO_COT_AAVE}
MATH_PROMPT_COT_SAE = "Question: {question}\nReasoning: "
MATH_PROMPT_COT_AAVE = "Question: {question}\nReasonin': "
MATH_PROMPT_COT = {"sae": MATH_PROMPT_COT_SAE, "aave": MATH_PROMPT_COT_AAVE}

MATH_INST = {"naive": MATH_INST_NAIVE, "cot": MATH_INST_COT}
MATH_INST_MQA = {"naive": MATH_INST_MQA_NAIVE, "cot": MATH_INST_MQA_COT}
MATH_DEMO = {"naive": MATH_DEMO_NAIVE, "cot": MATH_DEMO_COT}
MATH_PROMPT = {"naive": MATH_PROMPT_NAIVE, "cot": MATH_PROMPT_COT}


def format_prompts_math(template):
    def format_prompt(dataset, index):
        sample = dataset.iloc[index].to_dict() if isinstance(dataset, DataFrame) else dataset[index]
        if "{answer}" in template:
            if "{reasoning}" in template:
                formatted = template.format(question=sample["problem"], reasoning=sample["solution"], answer=sample["answer"])
            else:
                formatted = template.format(question=sample["problem"], answer=sample["answer"])
        else:
            formatted = template.format(question=sample["problem"])
        return formatted
    return format_prompt


# Planning
PLAN_INST_NAIVE_SAE = (
    "Solve the following planning problem. "
    "Assume that you need to execute all steps to complete the task and that infinite resources are available. "
    "What is the shortest possible time (in seconds) to complete this task? "
    """Your response should ONLY include your final answer in the format "#### {answer} seconds" (e.g.: #### 2 seconds)."""
)
PLAN_INST_NAIVE_AAVE = (
    "Solve the followin' plannin' problem. "
    "Assumin' you outta do all 'em steps to finish up the task, and you got infinite resources. "
    "What the shortest time (in seconds) be to knock this task out? "
    """You gotta respond with ONLY your final answer like "#### {answer} seconds" (somethin' like this: #### 2 seconds)."""
)
PLAN_INST_NAIVE = {"sae": PLAN_INST_NAIVE_SAE, "aave": PLAN_INST_NAIVE_AAVE}
PLAN_INST_MQA_NAIVE_SAE = (
    "Solve the following planning problem. "
    "Assume that you need to execute all steps to complete the task and that infinite resources are available. "
    "What is the shortest possible time (in seconds) to complete this task?\n{choices}\n\n"
    """Your response should ONLY include your final answer in the format "#### {{answer}}" (e.g.: #### C)."""
)
PLAN_INST_MQA_NAIVE_AAVE = (
    "Solve the followin' plannin' problem. "
    "Assumin' you outta do all 'em steps to finish up the task, and you got infinite resources. "
    "What the shortest time (in seconds) be to knock this task out?\n{choices}\n\n"
    """You gotta respond with ONLY your final answer like "#### {{answer}}" (somethin' like this: #### C)."""
)
PLAN_INST_MQA_NAIVE = {"sae": PLAN_INST_MQA_NAIVE_SAE, "aave": PLAN_INST_MQA_NAIVE_AAVE}
PLAN_FORMAT_NAIVE_SAE = """ONLY include your final answer in the format "#### {answer} seconds" (e.g.: #### 2 seconds)."""
# PLAN_FORMAT_NAIVE = {"sae": , "aave":}
PLAN_DEMO_NAIVE_SAE = "PROBLEM\n========\n{problem}\n#### {answer} seconds"
PLAN_DEMO_NAIVE_AAVE = "PROBLEM\n========\n{problem}\n#### {answer} seconds"
PLAN_DEMO_NAIVE = {"sae": PLAN_DEMO_NAIVE_SAE, "aave": PLAN_DEMO_NAIVE_AAVE}
PLAN_PROMPT_NAIVE_SAE = "PROBLEM\n========\n{problem}\n"
PLAN_PROMPT_NAIVE_AAVE = "PROBLEM\n========\n{problem}\n"
PLAN_PROMPT_NAIVE = {"sae": PLAN_PROMPT_NAIVE_SAE, "aave": PLAN_PROMPT_NAIVE_AAVE}

PLAN_INST_COT_SAE = (
    "Solve the following planning problem. "
    "Assume that you need to execute all steps to complete the task and that infinite resources are available. "
    "What is the shortest possible time (in seconds) to complete this task? "
    "Let's think step by step. "
    """End your response with your final answer in the format "#### {answer} seconds" (e.g.: #### 2 seconds)."""
)
PLAN_INST_COT_AAVE = (
    "Solve the followin' plannin' problem. "
    "Assumin' you outta do all 'em steps to finish up the task, and you got infinite resources. "
    "What the shortest time (in seconds) be to knock this task out? "
    "Aight, let's break it down step by step. "
    """You gotta end your answer with "#### {answer} seconds" (somethin' like this: #### 2 seconds)."""
)
PLAN_INST_COT = {"sae": PLAN_INST_COT_SAE, "aave": PLAN_INST_COT_AAVE}
PLAN_INST_MQA_COT_SAE = (
    "Solve the following planning problem. "
    "Let's think step by step. "
    "Assume that you need to execute all steps to complete the task and that infinite resources are available. "
    "What is the shortest possible time (in seconds) to complete this task?\n{choices}\n\n"
    """Your response should ONLY include your final answer in the format "#### {{answer}}" (e.g.: #### C)."""
)
PLAN_INST_MQA_COT_AAVE = (
    "Solve the followin' plannin' problem. "
    "Aight, let's break it down step by step. "
    "Assumin' you outta do all 'em steps to finish up the task, and you got infinite resources. "
    "What the shortest time (in seconds) be to knock this task out?\n{choices}\n\n"
    """You gotta respond with ONLY your final answer like "#### {{answer}}" (somethin' like this: #### C)."""
)
PLAN_INST_MQA_COT = {"sae": PLAN_INST_MQA_COT_SAE, "aave": PLAN_INST_MQA_COT_AAVE}
PLAN_DEMO_COT_SAE = "PROBLEM\n========\n{problem}\n\nReasoning: {reasoning}\n#### {answer} seconds"
PLAN_DEMO_COT_AAVE = "PROBLEM\n========\n{problem}\n\nReasonin': {reasoning}\n#### {answer} seconds"
PLAN_DEMO_COT = {"sae": PLAN_DEMO_COT_SAE, "aave": PLAN_DEMO_COT_AAVE}
PLAN_PROMPT_COT_SAE = "PROBLEM\n========\n{problem}\n\nReasoning: "
PLAN_PROMPT_COT_AAVE = "PROBLEM\n========\n{problem}\n\nReasonin': "
PLAN_PROMPT_COT = {"sae": PLAN_PROMPT_COT_SAE, "aave": PLAN_PROMPT_COT_AAVE}

PLAN_INST = {"naive": PLAN_INST_NAIVE, "cot": PLAN_INST_COT}
PLAN_INST_MQA = {"naive": PLAN_INST_MQA_NAIVE, "cot": PLAN_INST_MQA_COT}
PLAN_DEMO = {"naive": PLAN_DEMO_NAIVE, "cot": PLAN_DEMO_COT}
PLAN_PROMPT = {"naive": PLAN_PROMPT_NAIVE, "cot": PLAN_PROMPT_COT}


def format_prompts_planning(template):
    def format_prompt(dataset, index):
        sample = dataset.iloc[index].to_dict() if isinstance(dataset, DataFrame) else dataset[index]
        if "{answer}" in template:
            if "{reasoning}" in template:
                formatted = template.format(problem=sample["problem"], reasoning=sample["solution"], answer=sample["answer"])
            else:
                formatted = template.format(problem=sample["problem"], answer=sample["answer"])
        else:
            formatted = template.format(problem=sample["problem"])
        return formatted
    return format_prompt


# Summary
INSTS = {"algorithm": ALGO_INST, "logic": LOGIC_INST, "math": MATH_INST, "planning": PLAN_INST}
INSTS_MQA = {"algorithm": ALGO_INST_MQA, "logic": LOGIC_INST_MQA, "math": MATH_INST_MQA, "planning": PLAN_INST_MQA}
DEMOS = {"algorithm": ALGO_DEMO, "logic": LOGIC_DEMO, "math": MATH_DEMO, "planning": PLAN_DEMO}
PROMPTS = {"algorithm": ALGO_PROMPT, "logic": LOGIC_PROMPT, "math": MATH_PROMPT, "planning": PLAN_PROMPT}
FORMAT_PROMPTS_REGISTRY = {
    "algorithm": format_prompts_algorithm,
    "logic": format_prompts_logic,
    "math": format_prompts_math,
    "planning": format_prompts_planning
}
