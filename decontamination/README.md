# Decontamination

We use a 10-gram overlap to retrieve potentially contaminated samples, similarly to [Phi-1](https://huggingface.co/papers/2306.11644). 
After retrieving the candidates, we run a diff between the dataset sample and the benchmark sample using `difflib.SequenceMatcher` and discard the sample if `len(matched_substrings)/len(benchmark_sample) > 0.5`. 
We run decontamination against all the benchmarks we evaluated the Cosmo-1B model on: MMLU, HellaSwag, PIQA, SIQA, Winogrande, OpenBookQA, ARC-easy, ARC-challenge. 

[todo]: upload the code
