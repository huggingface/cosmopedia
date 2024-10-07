import argparse
from datasets import load_dataset
from vllm import LLM, SamplingParams


TEMPLATE = """Read the following web page content:
```
{}
```

Analyze the given web page content and determine if it contains clear mathematical deduction, reasoning, or step-by-step solutions suitable for a general audience. Suitable content includes:

- Clear mathematical deductions
- Logical reasoning in mathematical contexts
- Step-by-step solutions to mathematical problems

Do not select pages that:

- Are academic papers or use highly technical language
- Are irrelevant to mathematics
- Only contain mathematical formulas without explanation

Question-answer formats (e.g., from educational websites or forums) are acceptable if they meet the criteria. Ignore formatting errors or missing equations and make assumptions based on the overall content.

Provide a brief summary of the page with an explanation of your decision in 50 words or less. Conclude with "Verdict: select" if the content matches the criteria, or "Verdict: skip" if it doesn't.
"""

llm = LLM("meta-llama/Meta-Llama-3.1-70B-Instruct", download_dir="/scratch/cosmo/.cache/", tensor_parallel_size=4)
sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=256)


def label_text(batch):
    prompts = [TEMPLATE.format(text.strip()[:5000]) for text in batch["text"]]
    outputs = llm.generate(prompts, sampling_params)

    responses = []
    labels = []

    for output in outputs:
        response = output.outputs[0].text
        if "verdict: select" in response.lower():
            label = 1
        elif "verdict: skip" in response.lower():
            label = 0
        else:
            label = -1

        responses.append(response)
        labels.append(label)

    return {"response": responses, "label": labels}


def main(args):
    dataset = load_dataset(
        "parquet",
        data_files=f"{args.input_path}*.parquet",
        split="train",
        cache_dir="/scratch/cosmo/cache/",
        num_proc=1,
    )
    dataset = dataset.filter(
        lambda x, i: i % args.num_shards == args.shard, with_indices=True, num_proc=1
    )

    dataset = dataset.map(label_text, batched=True, batch_size=512, num_proc=1)
    dataset.to_parquet(f"{args.output_path}shard_{args.shard}.parquet")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path", type=str, default="s3://cosmopedia-data/re_extract_cc/llama_math/owm_candidates/"
    )
    parser.add_argument(
        "--output_path", type=str, default="s3://cosmopedia-data/re_extract_cc/llama_math/owm_llama_3_1_labels/"
    )
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num_shards", type=int, required=True)

    args = parser.parse_args()
    main(args)
