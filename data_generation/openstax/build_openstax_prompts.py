import pandas as pd
import random
import argparse
import numpy as np
from datasets import Dataset, load_dataset, concatenate_datasets


STYLES = {"young children":
{"beginning": "Create a fun and simple e-learning module on {{X}}, tailored for 5 to 10 year-old children. Opt for a playful and imaginative approach, suitable for very young learners.\n",
"criteria":"""In this module for young children, aim to:

- Use very simple, everyday words and phrases that a 5-year-old would easily understand, avoiding any complex concepts or technical terms.
- Tell a short, engaging story with colorful cartoon characters. For instance, to illustrate economic trade concepts use characters like animals or friendly creatures trading snacks or toys. Another example is addition and calculus, use apples to explain: '2 apples + 3 apples = 5 apples' .
- Keep the tone light, cheerful, and encouraging. Do not use images."""},

"middle school students":
{"beginning": "Create an engaging and accessible e-learning module on {{X}}, tailored for middle school students without prior knowledge on the topic.\n",
"criteria": """Instead of a traditional textbook approach, use a story-based narrative to explain the concept. Try to:

- Avoid technical jargon and present the ideas in a straightforward, conversational tone to spark curiosity and relate to the experiences of a younger audience.
- Include interactive elements like thought experiments and real-life scenarios. The goal is to topic approachable and fun, sparking curiosity about how it applies to everyday life.
- Do not use introductory phrases such as "welcome to this unit" at the beginning or conclusions the end. Do not use images."""},

"professionals and researchers":
{"beginning": "Create an extract of a scientific journal article for {{X}}, tailored for professionals and researchers on the topic.\n",
"criteria": """The style should mirror that of a scholarly publication, not school textbooks, aiming to engage a highly knowledgeable audience with very deep expertise. Try to:

- Present advanced theories, using technical and academic language.
- Include critical analysis of recent research findings and debates in the field, with a detailed examination of empirical data and statistical methodologies.
- The article should reflect the depth and complexity of content found in top-tier economics journals, intended for a readership deeply entrenched in the field.
- Do not add come up with references or add them at the end of the article. If there are mathematical expressions use a correct LateX formatting and do not use images."""},

"college students":
{"beginning": "Write a comprehensive and in-depth textbook on {{X}}, tailored for college students.\n",
"criteria": """Try to be:

- Rigorous: Ensure very detailed and in-depth coverage of the concepts.
- Engaging: Write with an academic and engaging tone that captivates interest.
- Applied: Use specific and practical examples. For example, if the topic is integration in calculus, include equations and proofs of the concept you're teaching. As another example, if the topic is the history of the United States, include dates, names, and key events.
If there are mathematical expressions use a correct LateX formatting. Do not use images and avoid introductory phrases such as "welcome to this unit" at the beginning or conclusions the end."""}}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="HuggingFaceTB/openstax_prompts")
    return parser.parse_args()


def parse_chapter(chapter, level=0, trail=[]):
    """Parse each chapter recursively and take into account sections"""
    trail_current = trail + [chapter["title"]]
    chapter_info = {
        "title": chapter["title"],
        "level": level,
        "trail": trail_current,
        "abstract": chapter.get("abstract", ""),
        "sections": [],
        "sub_chapters": [],
    }

    if chapter.get("sections"):
        for section in chapter.get("sections"):
            chapter_info["sections"].append(
                {
                    "title": section["title"],
                    "content": section.get("paragraph", ""),
                    "trail": trail_current,
                    "abstract": section.get("abstract", ""),
                }
            )

    # Handle sub-chapters recursively
    if chapter.get("chapters"):
        for sub_chapter in chapter.get("chapters"):
            chapter_info["sub_chapters"].append(
                parse_chapter(sub_chapter, level + 1, trail_current)
            )

    return chapter_info


def parse_book(book):
    """Parse and rearrange a book"""
    book_info = {"title": book["book_title"], "chapters": []}

    for chapter in book["chapters"]:
        if "preface" in chapter["title"].lower():
            continue
        book_info["chapters"].append(parse_chapter(chapter))

    return book_info


def build_prompts(
    parsed_book, style="college students", include_reference=True, refrence_size=500
):
    """Build prompts based on the (deepest) sections in each book"""
    prompts = []
    target_units = []
    chapters = [chap["title"] for chap in parsed_book["chapters"]]
    chosen_style = STYLES[style]
    start_prompt = chosen_style["beginning"]
    end_prompt = "\n" + chosen_style["criteria"]
    empty_content = 0
    
    for i, chapter in enumerate(parsed_book["chapters"]):
        chapter_prompt = start_prompt.replace("{{X}}", f"'{parsed_book['title']}'")
        chapter_prompt += f"We are writing on chapter '{chapter['title']}'. "

        for subchapter in chapter["sub_chapters"]:
            # Iterate over sections in subchapters
            if subchapter["sections"]:
                subchapter_prompt = (
                    chapter_prompt + f"In particular, section '{subchapter['title']}'. "
                )
                for i, unit in enumerate(subchapter["sections"]):
                    if i != 0:
                        units = [s["title"] for s in subchapter["sections"][:i]]
                        units = list(np.random.choice(units, random.randint(2, 5))) if len(units) > 5 else units
                        prev_units = ", ".join([f"'{name}'" for name in units])
                        plural = "s" if i > 1 else ""
                        subchapter_prompt += f"We have already covered the following unit{plural} in this section: {prev_units}. "
                    if unit["content"] and include_reference:
                        size = len(unit["content"])
                        ref = f" Here's some text for inspiration: {unit['content'][:min(refrence_size, size)]}".rstrip(".").rstrip() + "."
                    else:
                        empty_content += 1
                        ref = ""
                    new_prompt = (
                        subchapter_prompt
                        + f"Write a new unit titled '{unit['title']}'.{ref}\n"
                    )
                    prompts.append(new_prompt + end_prompt)
                    target_units.append(unit['title'])
            else:
                # Handle nested subchapters
                for k, e in enumerate(subchapter["sub_chapters"]):
                    if e["sections"]:
                        subchapter_prompt = (
                            chapter_prompt
                            + f"In particular, section '{e['title']}' of '{subchapter['title']}' part. "
                        )
                        for i, unit in enumerate(e["sections"]):
                            current_prompt = subchapter_prompt
                            if i != 0:
                                units = [s["title"] for s in e["sections"][:i]]
                                units = list(np.random.choice(units, random.randint(2, 5))) if len(units) > 5 else units
                                prev_units = ", ".join([f"'{name}'" for name in units])
                                plural = "s" if i > 1 else ""
                                current_prompt += f"We have already covered the following unit{plural} in this section: {prev_units}. "
                            if unit["content"] and include_reference:
                                size = len(unit["content"])
                                ref = f" Here's some text for inspiration: {unit['content'][:min(refrence_size, size)]}".rstrip(".").rstrip() + "."
                            else:
                                empty_content += 1
                                ref = ""
                            new_prompt = (
                                current_prompt
                                + f"Write a new unit titled '{unit['title']}'.{ref}\n"
                            )
                            target_units.append(unit['title'])
                            prompts.append(new_prompt + end_prompt)
                    else:
                        if "introduction" not in e['title'].lower() and e.get("abstract"):
                            new_prompt = chapter_prompt
                            if k != 0:
                                subchapters = [s["title"] for s in subchapter["sub_chapters"][:k]]
                                subchapters = list(np.random.choice(subchapters, random.randint(2, 5))) if len(subchapters) > 5 else subchapters
                                prev_subchapters = ", ".join([f"'{name}'" for name in subchapters])
                                plural = "s" if k > 1 else ""
                                new_prompt += f"We have already covered the following unit{plural} in this chapter: {prev_subchapters}. "
                            new_prompt = new_prompt + f"Write a new unit titled {e['title']}."
                            if include_reference:
                                size = len(e["abstract"])
                                new_prompt += f" Here's some text for inspiration: {e['abstract'][:min(refrence_size, size)]}"
                            else:
                                empty_content += 1
                            target_units.append(e['title'])
                            prompts.append(new_prompt.rstrip('.').rstrip() + '.\n' + end_prompt)
                        else:
                            continue
    return prompts, target_units, empty_content


if __name__ == "__main__":
    args = get_args()
    include_references = [True, False]
    refrence_size = 600
    ds = load_dataset("HuggingFaceTB/openstax_paragraphs", split="train")
    ds_en = ds.filter(lambda x: x["language"] == "en")

    print(f"English books dataset: {ds_en}")
    print("🔍 Parsing books...")
    parsed_books = [parse_book(e) for e in ds_en]
    datasets_list = []
    for include_reference in include_references:
        ref = "with_ref" if include_reference else "no_ref"
        for style in STYLES:
            print(f"🧩 Building prompts for {style} {ref}...")
            outputs = [
                build_prompts(
                    book,
                    style=style,
                    include_reference=include_reference,
                    refrence_size=refrence_size,
                )
                for book in parsed_books
            ]
            prompts = [p[0] for p in outputs]
            target_units = [p[1] for p in outputs]
            empty_content = [p[2] for p in outputs]
            sizes = [len(p) for p in prompts]
            print(
                f"✅ Done building {sum(sizes)} prompts! ({sum(empty_content)} without reference text)"
            )

            print(f"🌟 Examples:")
            print(f"- {prompts[random.randint(0, 5)][0]}")
            print(f"\n- {prompts[random.randint(0, 5)][-1]}")

            print("Converting to HF dataset and pushing to Hub...")
            flattened_prompts = []
            for book_prompts, book_units in zip(prompts, target_units):
                for prompt, unit in zip(book_prompts, book_units):
                    book_title = prompt.split(", tailored for")[0].split("'")[1]
                    flattened_prompts.append((prompt, unit, book_title))

            df = pd.DataFrame(flattened_prompts, columns=["prompt", "unit", "book title"])
            ds = Dataset.from_pandas(df)
            audience = "_".join(style.split(" "))
            ds = ds.add_column("audience", [f"{audience}_{ref}" for _ in range(len(ds))])
            print(ds)
            datasets_list.append(ds)

    final_ds = concatenate_datasets(datasets_list)
    print(final_ds)
    final_ds.push_to_hub(args.repo_id, private=True)
    


