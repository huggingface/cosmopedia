import argparse
import random
from datasets import load_dataset


STYLES = {"young_children_story":
"""Write an educational story (3-5 paragraphs) targeted at young children using simple words. The story should be inspired from this text snippet: 
“<EXTRACT>”

The story doesn’t have to be addressing everything in the snippet, it is there just for inspiration.
The story should have the following features: 
- Science integration: embed basic science concepts within the story, explaining them through the characters' adventures and discoveries. For example, if the story includes a scene where characters are looking at the sky, you could have them wonder why it's blue and explain the physics behind in grade-school level.
- Dialogue: include at least one dialogue and insightful conversation.
- Unexpected twist: conclude with a twist that doesn't resolve as hoped, but leaves a clear lesson about life and science.
Do not start with classic sentences like "Once upon a time", be creative.""",

"morality_story":
"""Write a compelling story related to the following text snippet:
“<EXTRACT>”

The story doesn’t need to mention everything in the snippet, use it just for inspiration and be creative!
The story should incorporate the following elements:
- Dialogue: the story must feature at least one meaningful dialogue that reveals character depth, advances the plot, or unravels a crucial piece of the mystery 
- Interesting themes: explore themes resonant with a mature audience, such as moral ambiguity, existential queries, personal transformation, or the consequences of past actions.
Do not start with classic sentences like "Once upon a time", "The sun hung low in the sky" or "In the dimly lit", be creative.""",

"problem_solving_story":
"""Write a story that explores a situation slightly related to this text snippet:
“<EXTRACT>”

The story should unfold through the characters interactions, decisions, and the consequences of their actions. Aim to weave in common sense lessons and social cues, emphasizing the importance of problem-solving. The narrative should cater to a diverse age group, including at least one dialogue and presenting both positive and negative outcomes.
Do not start with classic sentences like "Once upon a time", be creative.""",

"forums_story":
"""Write a story in the style of real-life situations that people share in forums. The story should be somehow related to this text snippet:
“<EXTRACT>”

The story needs to include a compelling and unexpected plot twist. Your narrative should resonate with the authenticity and personal touch found in forum discussions. Include relatable events and emotional depth.
Do not start with classic sentences like "Once upon a time", "A few years back" or "A few montsh ago", be creative.""",

"reddit_post":
"""Write a real-life story shared by someone in a reddit forum. The story should be somehow related to this text snippet: 
“<EXTRACT>”

The story should include: 
- Niche interests or humor: dive into specific hobbies, interests, or humorous situations 
- An unexpected plot twist or engaging conflict: introduce a relatable yet challenging situation or dilemma that the author faced. 
- Reflection and insight: end with a resolution that offers a new understanding, a sense of community, or a personal revelation, much like the conclusions drawn in forum discussions.
Start the story right away. Do not start with sentences like  "Once upon a time" as this is a reddit post and not a novel, you should also avoid starting with classic sentences like "A few years ago" or "A few years back", be creative."""}

EXTRACT_SIZE = 1000


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="HuggingFaceTB/ultrachat_stories_prompts")
    parser.add_argument("--generation_style", type=str, default="textbook_academic")
    parser.add_argument("--run_all_styles", action="store_true")
    return parser.parse_args()


def build_prompt(x, style="forums_story"):
    """Build the prompt based on the generation type"""
    snippet = x["first_turn"].strip()
    snippet = snippet[:min(len(snippet), EXTRACT_SIZE)]
    prompt = STYLES[style].replace("<EXTRACT>", snippet)
    return {f"prompt_{style}": prompt}


if __name__ == "__main__":
    args = get_args()

    print(f"Loading ultrachat data...")
    ds = load_dataset("HuggingFaceTB/ultrachat_questions_about_world", split="train")
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
    print(ds[0]["prompt_young_children_story"])
    print("-"*100)
    print(ds[1]["prompt_morality_story"])
    print("-"*100)
    print(ds[2]["prompt_reddit_post"])
    ds.push_to_hub(f"{args.repo_id}_{suffix}", private=True)
    print(f"✅ Data available at {args.repo_id}_{suffix}!")
