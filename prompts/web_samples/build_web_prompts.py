import argparse
import random
from datasets import load_dataset


STYLES = {"wikihow":
"""Here is an extract from a webpage: "<INSERT_EXTRACT>".

Write a long and very detailed tutorial that could be part of WikiHow whose title is related to the extract above<ADD_TOPIC>. Include in depth explanations for each step and how it helps achieve the desired outcome, inluding key tips and guidelines. 
Ensure clarity and practicality, allowing readers to easily follow and apply the instructions. Do not use images.""",

"textbook_narrative":
"""Here is an extract from a webpage: "<INSERT_EXTRACT>".

Write an extensive and detailed course unit suitable for a textbook, related to the given extract<ADD_TOPIC>. Do not just list concepts, but develop each one in detail before moving to the next, as we prioritize depth of understanding and comprehensive exploration of the subject matter over breadth. Focus on:

- Rigor: Ensure in-depth coverage of the concepts.
- Engagement: Use a narrative style akin to Michael Lewis, making it captivating and thought-provoking.
- Relevance: Connect the topic with current trends, real-life examples, or recent studies. Do not use images.
Do not include a title or an introduction, simply write the content without headlines and introductory phrases. Do not use images.""",

"textbook_academic":
"""Here is an extract from a webpage: "<INSERT_EXTRACT>".

Write an extensive and detailed course unit suitable for a textbook targeted at college students, related to the given extract<ADD_TOPIC>. Do not just list concepts, but develop each one in detail before moving to the next, as we prioritize depth of understanding and comprehensive exploration of the subject matter over breadth. Focus on:

- Rigor: Ensure in-depth coverage of the concepts/sections.
- Engagement: Write with an academic, professional and engaging tone that captivates interest.
- Application: Incorporate specific, practical examples, such as proofs in calculus or critical dates and figures in history.
Do not include a title or an introduction, simply write the content without headlines and introductory phrases. Do not use images.""",

"blogpost":
"""Here is an extract from a webpage: "<INSERT_EXTRACT>".

Write an informative and insightful blog post that expands upon the extract above<ADD_TOPIC>. Your post should delve into the nuances of the topic, offering fresh perspectives and deeper analysis. Aim to:

- Inform: Provide valuable, well-researched information that educates the reader.
- Engage: Write in a conversational tone that connects with the audience, making complex ideas accessible.
- Illustrate: Use examples, anecdotes, or personal experiences to bring the topic to life.
Do not give a title and do not start with sentences like "Have you ever..." or "Hello dear readers..", simply write the content without these introductory phrases."""
}

EXTRACT_SIZE = 1000


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="HuggingFaceTB/web_prompts")
    parser.add_argument("--data_type", type=str, default="textbook")
    parser.add_argument("--generation_style", type=str, default="textbook_academic")
    parser.add_argument("--run_all_styles", action="store_true")
    return parser.parse_args()


def build_prompt(x, style="textbook_academic"):
    """Build the prompt based on the generation type"""
    # web extract and topic
    web_sample = x["examples"]
    web_sample = web_sample[:min(EXTRACT_SIZE, len(web_sample))]
    topic = x["category"]
    add_topic = f', within the context of "{topic}"' if random.random() < 0.5 else ""
    # requested generation style
    prompt = STYLES[style].replace("<ADD_TOPIC>", add_topic).replace("<INSERT_EXTRACT>", web_sample)
    return {f"prompt_{style}": prompt}


if __name__ == "__main__":
    # load data=data_type and generate content in style=stayle
    args = get_args()

    print(f"Loading data fw2_as_{args.data_type}...")
    ds = load_dataset(f"HuggingFaceTB/fw2_as_{args.data_type}", split="train", num_proc=48)
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
    print(ds[0]["prompt_textbook_academic"])
    print("-"*100)
    print(ds[1]["prompt_textbook_academic"])
    print("-"*100)
    print(ds[2]["prompt_textbook_academic"])
    ds.push_to_hub(f"{args.repo_id}_{args.data_type}{suffix}", private=True)
    print(f"✅ Data available at {args.repo_id}_{args.data_type}{suffix}!")