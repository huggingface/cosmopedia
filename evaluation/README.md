# Benchmark evaluation

This is an extended version of the [FineWeb-v1 evaluation script](https://huggingface.co/datasets/HuggingFaceFW/fineweb/blob/main/lighteval_tasks.py)
In particular, we add the MMLU-Pro, TriviaQA, and GSM8k benchmarks.

To run the script, please install the latest version of the `lighteval` library:
```bash
git clone https://github.com/huggingface/lighteval.git
cd lighteval
# important to checkout the commit that contains a fix for the short context models
git checkout 60646959cea6ff183688a37eeae443efc1fa4584 
conda create -n lighteval python=3.10 && conda activate lighteval
pip install '.[accelerate,quantization,adapters]'
pip install deepspeed
```

Setup the acclerate using `accelerate config`. Then, you can run the evaluation script with the following command:
```bash
MODEL="openai-community/gpt2"
OUTPUT_DIR="results/"
accelerate launch --num_processes=1 --main_process_port=29600 run_evals_accelerate.py --model_args="pretrained=$MODEL" \
      --custom_tasks "lighteval_tasks.py" --output_dir $OUTPUT_DIR --override_batch_size 16 \
      --tasks "custom|hellaswag|0|1,custom|winogrande|0|1,custom|piqa|0|1,custom|siqa|0|1,custom|openbookqa|0|1,custom|arc:easy|0|1,custom|arc:challenge|0|1,custom|commonsense_qa|0|1,custom|trivia_qa|0|1,custom|mmlu_pro_cloze|0|1,custom|gsm8k|5|1,custom|mmlu_cloze:abstract_algebra|0|1,custom|mmlu_cloze:anatomy|0|1,custom|mmlu_cloze:astronomy|0|1,custom|mmlu_cloze:business_ethics|0|1,custom|mmlu_cloze:clinical_knowledge|0|1,custom|mmlu_cloze:college_biology|0|1,custom|mmlu_cloze:college_chemistry|0|1,custom|mmlu_cloze:college_computer_science|0|1,custom|mmlu_cloze:college_mathematics|0|1,custom|mmlu_cloze:college_medicine|0|1,custom|mmlu_cloze:college_physics|0|1,custom|mmlu_cloze:computer_security|0|1,custom|mmlu_cloze:conceptual_physics|0|1,custom|mmlu_cloze:econometrics|0|1,custom|mmlu_cloze:electrical_engineering|0|1,custom|mmlu_cloze:elementary_mathematics|0|1,custom|mmlu_cloze:formal_logic|0|1,custom|mmlu_cloze:global_facts|0|1,custom|mmlu_cloze:high_school_biology|0|1,custom|mmlu_cloze:high_school_chemistry|0|1,custom|mmlu_cloze:high_school_computer_science|0|1,custom|mmlu_cloze:high_school_european_history|0|1,custom|mmlu_cloze:high_school_geography|0|1,custom|mmlu_cloze:high_school_government_and_politics|0|1,custom|mmlu_cloze:high_school_macroeconomics|0|1,custom|mmlu_cloze:high_school_mathematics|0|1,custom|mmlu_cloze:high_school_microeconomics|0|1,custom|mmlu_cloze:high_school_physics|0|1,custom|mmlu_cloze:high_school_psychology|0|1,custom|mmlu_cloze:high_school_statistics|0|1,custom|mmlu_cloze:high_school_us_history|0|1,custom|mmlu_cloze:high_school_world_history|0|1,custom|mmlu_cloze:human_aging|0|1,custom|mmlu_cloze:human_sexuality|0|1,custom|mmlu_cloze:international_law|0|1,custom|mmlu_cloze:jurisprudence|0|1,custom|mmlu_cloze:logical_fallacies|0|1,custom|mmlu_cloze:machine_learning|0|1,custom|mmlu_cloze:management|0|1,custom|mmlu_cloze:marketing|0|1,custom|mmlu_cloze:medical_genetics|0|1,custom|mmlu_cloze:miscellaneous|0|1,custom|mmlu_cloze:moral_disputes|0|1,custom|mmlu_cloze:moral_scenarios|0|1,custom|mmlu_cloze:nutrition|0|1,custom|mmlu_cloze:philosophy|0|1,custom|mmlu_cloze:prehistory|0|1,custom|mmlu_cloze:professional_accounting|0|1,custom|mmlu_cloze:professional_law|0|1,custom|mmlu_cloze:professional_medicine|0|1,custom|mmlu_cloze:professional_psychology|0|1,custom|mmlu_cloze:public_relations|0|1,custom|mmlu_cloze:security_studies|0|1,custom|mmlu_cloze:sociology|0|1,custom|mmlu_cloze:us_foreign_policy|0|1,custom|mmlu_cloze:virology|0|1,custom|mmlu_cloze:world_religions|0|1"
```