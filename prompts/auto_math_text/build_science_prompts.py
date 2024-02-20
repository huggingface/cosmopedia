import argparse
from datasets import load_dataset


STYLES = {"college":
"""Write an educational piece suited for college students related to the following text snippet:
"<EXTRACT>"

Do not just list concepts, but develop each one in detail before moving to the next, as we prioritize depth of understanding and comprehensive exploration of the subject matter over breadth. Focus on:

- Rigor: Ensure in-depth coverage of the concepts/sections.
- Engagement: Write with an academic, professional and engaging tone that captivates interest.
- Application: Incorporate specific, practical examples, such as proofs in calculus or critical dates and figures in history.
Do not include a title or an introduction, simply write the content without headlines and introductory phrases. Do not use images.""",

"grade_school":
"""Here's an extract from a webpage:
"<EXTRACT>"

Create an educational piece related to the snippet above targeted at grade-school students. Complex college-like topics such Electromagnetism and Integration shouldn't be used, as they aren't usually taught at grade-school. If that's what the snippet is about, look for a much simpler scientific alternative to explain, and use everyday examples. For instance, if the topic is 'Linear Algebra' you might discuss how arranging objects in rows and columns can help solve puzzles.
Avoid technical terms and LaTeX and only discuss simple grade-school level topics. Start the educational piece right away."""}

EXTRACT_SIZE = 1000


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="HuggingFaceTB/auto_math")
    parser.add_argument("--generation_style", type=str, default="college")
    parser.add_argument("--run_all_styles", action="store_true")
    return parser.parse_args()


def build_prompt(x, style="college"):
    """Build the prompt based on the generation type"""
    snippet = x["text"].strip()
    snippet = snippet[:min(len(snippet), EXTRACT_SIZE)]
    prompt = STYLES[style].replace("<EXTRACT>", snippet)
    return {f"prompt_{style}": prompt}


if __name__ == "__main__":
    args = get_args()

    print(f"Loading AutoMathText web data...")
    ds = load_dataset("math-ai/AutoMathText", "web-0.50-to-1.00")["train"]
    if args.run_all_styles:
        suffix = ""
        for style in STYLES.keys():
            print(f"📖 Building prompts with a {style}...")
            ds = ds.map(build_prompt, num_proc=48, fn_kwargs={"style": style})
    else:
        suffix = f"_{args.generation_style}"
        print(f"📖 Building prompts with a {args.generation_style}...")
        ds = ds.map(build_prompt, num_proc=48, fn_kwargs={"style": args.generation_style})
        print(ds)
    print(ds)
    print(ds[0]["prompt_college"])
    print("-"*100)
    print(ds[1]["prompt_grade_school"])
    ds.push_to_hub(f"{args.repo_id}{suffix}", private=True)
    print(f"✅ Data available at {args.repo_id}{suffix}!")