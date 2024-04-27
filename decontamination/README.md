# Decontamination

We use a 10-gram overlap to retrieve potentially contaminated samples, similarly to [Phi-1](https://huggingface.co/papers/2306.11644). 
After retrieving the candidates, we run a diff between the dataset sample and the benchmark sample using `difflib.SequenceMatcher` and discard the sample if `len(matched_substrings)/len(benchmark_sample) > 0.5`. 
We run decontamination against all the benchmarks we evaluated the Cosmo-1B model on: MMLU, HellaSwag, PIQA, SIQA, Winogrande, OpenBookQA, ARC-easy, ARC-challenge. 

Usage:
```bash
export HF_DATASETS_CACHE=/scratch/cosmo/cache
export HUGGINGFACE_HUB_CACHE=/scratch/cosmo/cache

python decontaminate.py --train_dataset "HuggingFaceTB/AMT_2M_Khanacademy_24k" --report_dataset_name "HuggingFaceTB/AMT_2M_Khanacademy_24k_decont_report" --save_decontaminated --decontaminated_dataset_name "HuggingFaceTB/AMT_2M_Khanacademy_24k_decont"
```


