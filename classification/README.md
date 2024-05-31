# Educational value classifier

### 1. Finetune a model for educational value regression

* edit `train_edu_bert.slurm`
```bash
--base_model_name="Snowflake/snowflake-arctic-embed-m" \  # BERT-like base model
--dataset_name="HuggingFaceTB/LLM_juries_fineweb_430k_annotations" \  # Llama3-annotated eduational value dataset
--target_column="score" 
```
* run the training script on a SLURM cluster:
```bash
sbatch train_edu_bert.slurm
```

### 2. Annotate a dataset with the educational scores predicted by the model
    
```bash
sbatch run_edu_bert.slurm
```