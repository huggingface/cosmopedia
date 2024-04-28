# Setup

```bash
# run embeddings on data chunks
bash launch_embeddings.sh
# run classification using the embeddings and a trained classifier
bash launch_classification.sh
# merge the data chunks
python /fsx/loubna/projects/cosmopedia/prompts/judge/merge_data.py
```