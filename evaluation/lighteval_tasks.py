# ruff: noqa: F405, F403, F401
"""
Custom evaluation tasks for lighteval

Do note that we ran the evals with `max_samples=1000` to speed up large evals.
Most custom prompt changes were in an attempt to improve signal for small models in general.

This file generally creates just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.

Example usage (lighteval_tasks.py is the path to this file):
===================
accelerate launch --num_processes=1 lighteval/run_evals_accelerate.py --model_args="pretrained=HuggingFaceTB/cosmo-1b" \
    --custom_tasks "lighteval_tasks.py" --output_dir [OUTPUTPATH] --max_samples 1000 \
    --tasks "custom|hellaswag|0|1,custom|winogrande|0|1,custom|piqa|0|1,custom|siqa|0|1,custom|openbookqa|0|1,custom|arc:easy|0|1,custom|arc:challenge|0|1,custom|commonsense_qa|0|1,custom|mmlu:abstract_algebra|0|1,custom|mmlu:anatomy|0|1,custom|mmlu:astronomy|0|1,custom|mmlu:business_ethics|0|1,custom|mmlu:clinical_knowledge|0|1,custom|mmlu:college_biology|0|1,custom|mmlu:college_chemistry|0|1,custom|mmlu:college_computer_science|0|1,custom|mmlu:college_mathematics|0|1,custom|mmlu:college_medicine|0|1,custom|mmlu:college_physics|0|1,custom|mmlu:computer_security|0|1,custom|mmlu:conceptual_physics|0|1,custom|mmlu:econometrics|0|1,custom|mmlu:electrical_engineering|0|1,custom|mmlu:elementary_mathematics|0|1,custom|mmlu:formal_logic|0|1,custom|mmlu:global_facts|0|1,custom|mmlu:high_school_biology|0|1,custom|mmlu:high_school_chemistry|0|1,custom|mmlu:high_school_computer_science|0|1,custom|mmlu:high_school_european_history|0|1,custom|mmlu:high_school_geography|0|1,custom|mmlu:high_school_government_and_politics|0|1,custom|mmlu:high_school_macroeconomics|0|1,custom|mmlu:high_school_mathematics|0|1,custom|mmlu:high_school_microeconomics|0|1,custom|mmlu:high_school_physics|0|1,custom|mmlu:high_school_psychology|0|1,custom|mmlu:high_school_statistics|0|1,custom|mmlu:high_school_us_history|0|1,custom|mmlu:high_school_world_history|0|1,custom|mmlu:human_aging|0|1,custom|mmlu:human_sexuality|0|1,custom|mmlu:international_law|0|1,custom|mmlu:jurisprudence|0|1,custom|mmlu:logical_fallacies|0|1,custom|mmlu:machine_learning|0|1,custom|mmlu:management|0|1,custom|mmlu:marketing|0|1,custom|mmlu:medical_genetics|0|1,custom|mmlu:miscellaneous|0|1,custom|mmlu:moral_disputes|0|1,custom|mmlu:moral_scenarios|0|1,custom|mmlu:nutrition|0|1,custom|mmlu:philosophy|0|1,custom|mmlu:prehistory|0|1,custom|mmlu:professional_accounting|0|1,custom|mmlu:professional_law|0|1,custom|mmlu:professional_medicine|0|1,custom|mmlu:professional_psychology|0|1,custom|mmlu:public_relations|0|1,custom|mmlu:security_studies|0|1,custom|mmlu:sociology|0|1,custom|mmlu:us_foreign_policy|0|1,custom|mmlu:virology|0|1,custom|mmlu:world_religions|0|1"
===================

More info here: https://github.com/huggingface/lighteval?tab=readme-ov-file#evaluate-a-model-on-extended-community-or-custom-tasks
For more info on differences between MMLU implementations: https://huggingface.co/blog/open-llm-leaderboard-mmlu#1001-flavors-of-mmlu
In particular, the default leaderboard MMLU implementation (which uses "A", "B", etc as answer targets) gives generally random results on small/non instruction tuned models.
Instead, we use the full MMLU answer as the target.
"""
import re
from typing import List, Tuple

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.default_prompts import LETTER_INDICES

_TASKS_STRINGS: List[Tuple[LightevalTaskConfig, str]] = []
_TASKS: List[LightevalTaskConfig] = []

## COMMON_SENSE_REASONING_TASKS ##
COMMON_SENSE_REASONING_TASKS = [
    LightevalTaskConfig(
        name="hellaswag",
        prompt_function="hellaswag_prompt",
        hf_repo="hellaswag",
        hf_subset="default",
        metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    ),
    LightevalTaskConfig(
        name="winogrande",
        prompt_function="winogrande",
        hf_repo="winogrande",
        hf_subset="winogrande_xl",
        metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    ),
    LightevalTaskConfig(
        name="piqa",
        prompt_function="piqa_harness",
        hf_repo="piqa",
        hf_subset="plain_text",
        metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    ),
    LightevalTaskConfig(
        name="siqa",
        prompt_function="siqa_prompt",
        hf_repo="lighteval/siqa",
        hf_subset="default",
        hf_avail_splits=["train", "validation"],
        metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    ),
    LightevalTaskConfig(
        name="openbookqa",
        prompt_function="openbookqa",
        hf_repo="openbookqa",
        hf_subset="main",
        metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    ),
    LightevalTaskConfig(
        name="arc:easy",
        prompt_function="arc",
        hf_repo="ai2_arc",
        hf_subset="ARC-Easy",
        evaluation_splits=["test"],
        generation_size=1,
        metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    ),
    LightevalTaskConfig(
        name="arc:challenge",
        prompt_function="arc",
        hf_repo="ai2_arc",
        hf_subset="ARC-Challenge",
        evaluation_splits=["test"],
        generation_size=1,
        metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    ),
    LightevalTaskConfig(
        name="commonsense_qa",
        prompt_function="commonsense_qa_prompt",
        hf_repo="commonsense_qa",
        hf_subset="default",
        metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    ),
    LightevalTaskConfig(
        name="mmlu_pro_cloze",
        prompt_function="mmlu_pro_cloze_prompt",
        hf_repo="TIGER-Lab/MMLU-Pro",
        hf_subset="default",
        metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
        evaluation_splits=["test"],
        few_shots_split="validation",
        few_shots_select=None,
        generation_size=-1,
        stop_sequence=None,
        output_regex=None,
        frozen=False,
    ),
    LightevalTaskConfig(
        name="mmlu_pro_mc",
        prompt_function="mmlu_pro_mc_prompt",
        hf_repo="TIGER-Lab/MMLU-Pro",
        hf_subset="default",
        metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
        evaluation_splits=["test"],
        few_shots_split="validation",
        few_shots_select=None,
        generation_size=1,
        stop_sequence=None,
        output_regex=None,
        frozen=False,
    ),
    LightevalTaskConfig(
        name="boolq",
        prompt_function="boolq_prompt",
        hf_repo="super_glue",
        hf_subset="boolq",
        metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
        trust_dataset=True,
        stop_sequence=["\n"],
    ),
    LightevalTaskConfig(
        name="trivia_qa",
        prompt_function="triviaqa",
        hf_repo="mandarjoshi/trivia_qa",
        hf_subset="rc.nocontext",
        hf_avail_splits=["train", "validation"],
        evaluation_splits=["validation"],
        metric=[Metrics.quasi_exact_match_triviaqa],
        generation_size=20,
        trust_dataset=True,
        stop_sequence=["\n", ".", ","],
        few_shots_select="random_sampling_from_train",
    ),
]


def boolq_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"{line['passage']}\nQuestion: {line['question'].capitalize()}?\nAnswer:",
        choices=[" No", " Yes"],  # Only gold
        gold_index=int(line["label"]),
    )


def mmlu_pro_cloze_prompt(line, task_name: str = None):
    """MMLU-Pro prompt without letters"""
    topic = line["category"]
    prompt = f"The following are questions about {topic.replace('_', ' ')}.\nQuestion: "
    prompt += line["question"] + "\nAnswer:"

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[f" {c}" for c in line["options"]],
        gold_index=line["answer_index"],
        instruction=f"The following are questions about {topic.replace('_', ' ')}.\n",
    )


def mmlu_pro_mc_prompt(line, task_name: str = None):
    topic = line["category"]
    query = f"The following are multiple choice questions (with answers) about {topic.replace('_', ' ')}.\n\n"
    query += line["question"] + "\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, line["options"])])
    query += "Answer:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=LETTER_INDICES[: len(line["options"])],
        gold_index=line["answer_index"],
        instruction=f"The following are multiple choice questions (with answers) about {topic.replace('_', ' ')}.\n\n",
        target_for_fewshot_sorting=LETTER_INDICES[line["answer_index"]],
    )


def commonsense_qa_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["question"],
        choices=[f" {c}" for c in line["choices"]["text"]],
        gold_index=LETTER_INDICES.index(line["answerKey"].strip()),
        instruction="",
    )


def siqa_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["context"] + " " + line["question"],
        choices=[f" {c}" for c in [line["answerA"], line["answerB"], line["answerC"]]],
        gold_index=int(line["label"]) - 1,
        instruction="",
    )


def hellaswag_prompt(line, task_name: str = None):
    def preprocess(text):
        """Comes from AiHarness"""
        # text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    ctx = f"{line['ctx_a']} {line['ctx_b'].capitalize()} "
    return Doc(
        task_name=task_name,
        query=preprocess(line["activity_label"] + ": " + ctx),
        choices=[" " + preprocess(ending) for ending in line["endings"]],
        gold_index=int(line["label"]) if line["label"] != "" else -1,  # -1 for test
        # "metric": "choices_loglikelihood",
    )


GSM8K = LightevalTaskConfig(
    name="gsm8k",
    prompt_function="gsm8k",
    hf_repo="gsm8k",
    hf_subset="main",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    metric=[Metrics.quasi_exact_match_gsm8k],
    generation_size=256,
    stop_sequence=["Question:", "Question"],
    few_shots_select="random_sampling_from_train",
)
MATH_TASKS = [
    LightevalTaskConfig(
        name=f"math:{subset}",
        prompt_function="math",
        hf_repo="lighteval/MATH",
        hf_subset=subset,
        hf_avail_splits=["train", "test"],
        evaluation_splits=["test"],
        metric=[Metrics.quasi_exact_match_math],
        generation_size=256,
        stop_sequence=["Problem:", "Problem"],
        few_shots_select="random_sampling_from_train",
    )
    for subset in [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
]

# 0 short for common sense
COMMON_SENSE_REASONING_STRING = [(t, f"custom|{t.name}|0|1") for t in COMMON_SENSE_REASONING_TASKS]
_TASKS_STRINGS.extend(COMMON_SENSE_REASONING_STRING)
_TASKS_STRINGS.extend([(GSM8K, f"custom|{GSM8K.name}|5|1")])
_TASKS_STRINGS.extend([(t, f"custom|{t.name}|4|1") for t in MATH_TASKS])
_TASKS += COMMON_SENSE_REASONING_TASKS
_TASKS += [GSM8K] + MATH_TASKS

## MMLU ##
class CustomMMLUEvaluationTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        prompt_function="mmlu_prompt",
        hf_repo="lighteval/mmlu",
        hf_subset=None,
        #  metric=[Metrics.loglikelihood_acc_single_token],
        metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
        hf_avail_splits=None,
        evaluation_splits=["test"],
        few_shots_split="dev",
        few_shots_select=None,
        generation_size=-1,
        stop_sequence=None,
        output_regex=None,
        frozen=False,
    ):
        super().__init__(
            name=name,
            prompt_function=prompt_function,
            hf_repo=hf_repo,
            hf_subset=hf_subset,
            metric=metric,
            hf_avail_splits=hf_avail_splits,
            evaluation_splits=evaluation_splits,
            few_shots_split=few_shots_split,
            few_shots_select=few_shots_select,
            generation_size=generation_size,
            stop_sequence=stop_sequence,
            output_regex=output_regex,
            frozen=frozen,
        )

MMLU_TASKS = []
mmlu_subsets = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

for answer_type in ("mc", "cloze"):
    prompt_function = f"mmlu_{answer_type}_prompt"
    generation_size = -1 if answer_type == "cloze" else 1
    for subset in mmlu_subsets:
        MMLU_TASKS.append(
            CustomMMLUEvaluationTask(
                name=f"mmlu_{answer_type}:{subset}",
                prompt_function=prompt_function,
                hf_subset=subset,
                generation_size=generation_size
            )
        )

MMLU_TASKS += [
    CustomMMLUEvaluationTask(
        name=f"mmlu_stem_mc",
        hf_repo="TIGER-Lab/MMLU-STEM",
        prompt_function="mmlu_mc_prompt",
        hf_subset="default",
        generation_size=1
    ),
    CustomMMLUEvaluationTask(
        name=f"mmlu_stem_cloze",
        hf_repo="TIGER-Lab/MMLU-STEM",
        prompt_function="mmlu_cloze_prompt",
        hf_subset="default",
        generation_size=-1
    ),
]


def mmlu_cloze_prompt(line, task_name: str = None):
    """MMLU prompt without letters"""
    topic = line["subject"]
    prompt = f"The following are questions about {topic.replace('_', ' ')}.\nQuestion: "
    prompt += line["question"] + "\nAnswer:"

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[f" {c}" for c in line["choices"]],
        gold_index=line["answer"],
        instruction=f"The following are questions about {topic.replace('_', ' ')}.\n",
    )


def mmlu_mc_prompt(line, task_name: str = None):
    topic = line["subject"]
    query = f"The following are multiple choice questions (with answers) about {topic.replace('_', ' ')}.\n\n"
    query += line["question"] + "\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, line["choices"])])
    query += "Answer:"

    gold_ix = LETTER_INDICES.index(line["answer"]) if isinstance(line["answer"], str) else line["answer"]

    return Doc(
        task_name=task_name,
        query=query,
        choices=[" A", " B", " C", " D"],
        gold_index=gold_ix,
        instruction=f"The following are multiple choice questions (with answers) about {topic.replace('_', ' ')}.\n\n",
        target_for_fewshot_sorting=[" A", " B", " C", " D"][gold_ix],
    )


MMLU_STRING = [(t, f"custom|{t.name}|0|1") for t in MMLU_TASKS]
_TASKS_STRINGS.extend(MMLU_STRING)
_TASKS += MMLU_TASKS

# common sense reasoning + mmlu
EARLY_SIGNAL_TASKS = ",".join([t[1] for t in COMMON_SENSE_REASONING_STRING] + [t[1] for t in MMLU_STRING])

# Convert to dict for lighteval
TASKS_TABLE = _TASKS
# You can have a few pre-organised groups of tasks
TASKS_GROUPS = {
    "early-signal": EARLY_SIGNAL_TASKS,
    "math": f"custom|{GSM8K.name}|5|1" + "," + ",".join([f"custom|{t.name}|4|1" for t in MATH_TASKS]),
}
