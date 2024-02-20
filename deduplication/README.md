# Deduplication

We run deduplication on the dataset using MinHash from [datatrove](https://github.com/huggingface/datatrove). 
Considering that the seed samples had already undergone deduplication, and we carefully crafted the prompts to ensure distinct outputs even with identical seeds, the volume of duplicates found in Cosmopedia was less than 1% of the files, which were subsequenlty removed.

The deduplication script is available at `deduplicate_dataset.py`, make sure to follow the installation guidelines in `datatrove` and to change the paths in the file before running it.