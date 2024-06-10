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

from lighteval.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks_prompt_formatting import LETTER_INDICES

_TASKS_STRINGS: List[Tuple[LightevalTaskConfig, str]] = []
_TASKS: List[LightevalTaskConfig] = []

## COMMON_SENSE_REASONING_TASKS ##
COMMON_SENSE_REASONING_TASKS = [
    LightevalTaskConfig(
        name="hellaswag",
        prompt_function="hellaswag_prompt",
        hf_repo="hellaswag",
        hf_subset="default",
        metric=["loglikelihood_acc", "loglikelihood_acc_norm_nospace"],
    ),
    LightevalTaskConfig(
        name="winogrande",
        prompt_function="winogrande",
        hf_repo="winogrande",
        hf_subset="winogrande_xl",
        metric=["loglikelihood_acc", "loglikelihood_acc_norm_nospace"],
    ),
    LightevalTaskConfig(
        name="piqa",
        prompt_function="piqa_harness",
        hf_repo="piqa",
        hf_subset="plain_text",
        metric=["loglikelihood_acc", "loglikelihood_acc_norm_nospace"],
    ),
    LightevalTaskConfig(
        name="siqa",
        prompt_function="siqa_prompt",
        hf_repo="lighteval/siqa",
        hf_subset="default",
        hf_avail_splits=["train", "validation"],
        metric=["loglikelihood_acc", "loglikelihood_acc_norm_nospace"],
    ),
    LightevalTaskConfig(
        name="openbookqa",
        prompt_function="openbookqa",
        hf_repo="openbookqa",
        hf_subset="main",
        metric=["loglikelihood_acc", "loglikelihood_acc_norm_nospace"],
    ),
    LightevalTaskConfig(
        name="arc:easy",
        prompt_function="arc",
        hf_repo="ai2_arc",
        hf_subset="ARC-Easy",
        evaluation_splits=["test"],
        generation_size=1,
        metric=["loglikelihood_acc", "loglikelihood_acc_norm_nospace"],
    ),
    LightevalTaskConfig(
        name="arc:challenge",
        prompt_function="arc",
        hf_repo="ai2_arc",
        hf_subset="ARC-Challenge",
        evaluation_splits=["test"],
        generation_size=1,
        metric=["loglikelihood_acc", "loglikelihood_acc_norm_nospace"],
    ),
    LightevalTaskConfig(
        name="commonsense_qa",
        prompt_function="commonsense_qa_prompt",
        hf_repo="commonsense_qa",
        hf_subset="default",
        metric=["loglikelihood_acc", "loglikelihood_acc_norm_nospace"],
    ),
    LightevalTaskConfig(
        name="gsm8k",
        prompt_function="gsm8k",
        hf_repo="gsm8k",
        hf_subset="main",
        evaluation_splits=["test"],
        metric=["quasi_exact_match_gsm8k"],
        generation_size=256,
        stop_sequence=["Question:", "Question", ":"],
    )
]


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


# 0 short for common sense
COMMON_SENSE_REASONING_STRING = [(t, f"custom|{t.name}|0|1") for t in COMMON_SENSE_REASONING_TASKS]
_TASKS_STRINGS.extend(COMMON_SENSE_REASONING_STRING)
_TASKS += COMMON_SENSE_REASONING_TASKS

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
        suite=None,
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
            suite=suite,
            generation_size=generation_size,
            stop_sequence=stop_sequence,
            output_regex=output_regex,
            frozen=frozen,
        )


MMLU_TASKS = [
    CustomMMLUEvaluationTask(name="mmlu:abstract_algebra", hf_subset="abstract_algebra"),
    CustomMMLUEvaluationTask(name="mmlu:anatomy", hf_subset="anatomy"),
    CustomMMLUEvaluationTask(name="mmlu:astronomy", hf_subset="astronomy"),
    CustomMMLUEvaluationTask(name="mmlu:business_ethics", hf_subset="business_ethics"),
    CustomMMLUEvaluationTask(name="mmlu:clinical_knowledge", hf_subset="clinical_knowledge"),
    CustomMMLUEvaluationTask(name="mmlu:college_biology", hf_subset="college_biology"),
    CustomMMLUEvaluationTask(name="mmlu:college_chemistry", hf_subset="college_chemistry"),
    CustomMMLUEvaluationTask(name="mmlu:college_computer_science", hf_subset="college_computer_science"),
    CustomMMLUEvaluationTask(name="mmlu:college_mathematics", hf_subset="college_mathematics"),
    CustomMMLUEvaluationTask(name="mmlu:college_medicine", hf_subset="college_medicine"),
    CustomMMLUEvaluationTask(name="mmlu:college_physics", hf_subset="college_physics"),
    CustomMMLUEvaluationTask(name="mmlu:computer_security", hf_subset="computer_security"),
    CustomMMLUEvaluationTask(name="mmlu:conceptual_physics", hf_subset="conceptual_physics"),
    CustomMMLUEvaluationTask(name="mmlu:econometrics", hf_subset="econometrics"),
    CustomMMLUEvaluationTask(name="mmlu:electrical_engineering", hf_subset="electrical_engineering"),
    CustomMMLUEvaluationTask(name="mmlu:elementary_mathematics", hf_subset="elementary_mathematics"),
    CustomMMLUEvaluationTask(name="mmlu:formal_logic", hf_subset="formal_logic"),
    CustomMMLUEvaluationTask(name="mmlu:global_facts", hf_subset="global_facts"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_biology", hf_subset="high_school_biology"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_chemistry", hf_subset="high_school_chemistry"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_computer_science", hf_subset="high_school_computer_science"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_european_history", hf_subset="high_school_european_history"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_geography", hf_subset="high_school_geography"),
    CustomMMLUEvaluationTask(
        name="mmlu:high_school_government_and_politics", hf_subset="high_school_government_and_politics"
    ),
    CustomMMLUEvaluationTask(name="mmlu:high_school_macroeconomics", hf_subset="high_school_macroeconomics"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_mathematics", hf_subset="high_school_mathematics"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_microeconomics", hf_subset="high_school_microeconomics"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_physics", hf_subset="high_school_physics"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_psychology", hf_subset="high_school_psychology"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_statistics", hf_subset="high_school_statistics"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_us_history", hf_subset="high_school_us_history"),
    CustomMMLUEvaluationTask(name="mmlu:high_school_world_history", hf_subset="high_school_world_history"),
    CustomMMLUEvaluationTask(name="mmlu:human_aging", hf_subset="human_aging"),
    CustomMMLUEvaluationTask(name="mmlu:human_sexuality", hf_subset="human_sexuality"),
    CustomMMLUEvaluationTask(name="mmlu:international_law", hf_subset="international_law"),
    CustomMMLUEvaluationTask(name="mmlu:jurisprudence", hf_subset="jurisprudence"),
    CustomMMLUEvaluationTask(name="mmlu:logical_fallacies", hf_subset="logical_fallacies"),
    CustomMMLUEvaluationTask(name="mmlu:machine_learning", hf_subset="machine_learning"),
    CustomMMLUEvaluationTask(name="mmlu:management", hf_subset="management"),
    CustomMMLUEvaluationTask(name="mmlu:marketing", hf_subset="marketing"),
    CustomMMLUEvaluationTask(name="mmlu:medical_genetics", hf_subset="medical_genetics"),
    CustomMMLUEvaluationTask(name="mmlu:miscellaneous", hf_subset="miscellaneous"),
    CustomMMLUEvaluationTask(name="mmlu:moral_disputes", hf_subset="moral_disputes"),
    CustomMMLUEvaluationTask(name="mmlu:moral_scenarios", hf_subset="moral_scenarios"),
    CustomMMLUEvaluationTask(name="mmlu:nutrition", hf_subset="nutrition"),
    CustomMMLUEvaluationTask(name="mmlu:philosophy", hf_subset="philosophy"),
    CustomMMLUEvaluationTask(name="mmlu:prehistory", hf_subset="prehistory"),
    CustomMMLUEvaluationTask(name="mmlu:professional_accounting", hf_subset="professional_accounting"),
    CustomMMLUEvaluationTask(name="mmlu:professional_law", hf_subset="professional_law"),
    CustomMMLUEvaluationTask(name="mmlu:professional_medicine", hf_subset="professional_medicine"),
    CustomMMLUEvaluationTask(name="mmlu:professional_psychology", hf_subset="professional_psychology"),
    CustomMMLUEvaluationTask(name="mmlu:public_relations", hf_subset="public_relations"),
    CustomMMLUEvaluationTask(name="mmlu:security_studies", hf_subset="security_studies"),
    CustomMMLUEvaluationTask(name="mmlu:sociology", hf_subset="sociology"),
    CustomMMLUEvaluationTask(name="mmlu:us_foreign_policy", hf_subset="us_foreign_policy"),
    CustomMMLUEvaluationTask(name="mmlu:virology", hf_subset="virology"),
    CustomMMLUEvaluationTask(name="mmlu:world_religions", hf_subset="world_religions"),
]


def mmlu_prompt(line, task_name: str = None):
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


MMLU_STRING = [(t, f"custom|{t.name}|0|1") for t in MMLU_TASKS]
_TASKS_STRINGS.extend(MMLU_STRING)
_TASKS += MMLU_TASKS

# common sense reasoning + mmlu
EARLY_SIGNAL_TASKS = ",".join([t[1] for t in COMMON_SENSE_REASONING_STRING] + [t[1] for t in MMLU_STRING])

# Convert to dict for lighteval
TASKS_TABLE = [task.as_dict() for task in _TASKS]
# You can have a few pre-organised groups of tasks
TASKS_GROUPS = {
    "early-signal": EARLY_SIGNAL_TASKS,
}
