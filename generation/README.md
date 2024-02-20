
# Synthetic data generation

If you have a large dataset of prompts and want to generate content using an Open-Source LLM like [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1), you can use `llm-swarm` which spins TGI or vLLM instances on `slurm`clutsres, we used it to generated Cosmopedia, which con sists of 25B tokens. The full generation took around 16k H100 GPU hours.

You can find the instructions for running the generation in `llm-swarm` here: https://github.com/huggingface/llm-swarm/tree/loubna/examples/textbooks

The generation script is also available here (to be used within `examples/textbooks` of the library.)

```bash
# after having followed all the installation guidlines in llm-swrarm + install wandb
# 100k subset
python generate_syntehtic_textbooks.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --instances 2 \
    --prompts_dataset "HuggingFaceTB/cosmopedia-100k" \
    --prompt_column prompt \
    --max_samples 2000 \
    --checkpoint_path "./synthetic_data" \
    --checkpoint_interval 1000
```
