from datasets import load_dataset

BASE_PROMPT = """Here is an extract from a python coding tutorial: 
```
{code_snippet}
```
{audience_prompt} The textbook should promote reasoning and algorithmical skills. Do not just list concepts, but develop each one in detail before moving to the next, as we prioritize depth of understanding and comprehensive exploration of the subject matter over breadth.
Try to:
- Ensure in-depth coverage of the concepts.
- Use a narrative thought-provoking style.
- Use LaTeX notation $$ for equations and ``` for Python code snippets. 
- Ensure valid Markdown output.
- Install and import any necessary libraries.
Do not include a title, introductory phrases or images. Do not explain basic python concepts like functions and variables. Do not use html for formatting. Write the content directly."""

def add_prompts(example):
    middle_school_prompt = BASE_PROMPT.format(
            code_snippet=example["text"].strip()[-1000:],
            audience_prompt=(
                "Write an extensive and detailed textbook unit with interleaved text and code snippets for "
                "middle school students related to the extract above. Ensure the explanations are accessible "
                "and easy to understand by students with no prior knowledge of the subject."
            )
    )
    college_prompt = BASE_PROMPT.format(
            code_snippet=example["text"].strip()[-1000:],
            audience_prompt=(
                "Write an extensive and detailed textbook with interleaved text and code snippets for "
                "college students related to the extract above. Ensure the explanations are accessible "
                "and easy to understand by students with some basic knowledge of the subject. "
            )
    )

    return {
        "prompt_middle_school": middle_school_prompt,
        "prompt_college": college_prompt
    }



if __name__ == "__main__":
    data = load_dataset("math-ai/AutoMathText", "code-python-0.50-to-1.00", split="train",
                        cache_dir="/scratch/cosmo/cache", num_proc=32)
    data = data.map(add_prompts, num_proc=32)
    data.push_to_hub("HuggingFaceTB/prompts_code_textbooks_amt_python", private=True)

