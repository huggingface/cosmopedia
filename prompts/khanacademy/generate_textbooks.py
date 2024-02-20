import json
from string import Template
import pandas as pd

TEMPLATE = Template("""Write a long and very detailed course unit for a textbook on "${unit_title}".
${previous_sections}\"${section}\".
${previous_sub_units}
Write the new sub-unit titled \"${unit}\" while trying to be:
- Rigorous - you create challenging textbooks that cover the material in depth.
- Engaging - your textbooks have a narrative arc and engaging tone, like the writing of Michael Lewis.
- Applied - you use specific and practical examples. For example, if the topic is integration in calculus, include equations and proofs of the concept you're teaching. As another example, if the topic is the history of the United States, include dates, names, and key events.
Model:""")


# file created by khan_dl
with open("khan_courses.json") as f:
    data = json.load(f)

textbooks = []
total_courses = 0
total_sections = 0
for course in data:
    total_courses += 1
    units = course["subunits"]
    for ui, unit in enumerate(units):
        for sui, subunit in enumerate(unit["subunits"]):
            total_sections += 1
            for li, lesson_title in enumerate(subunit["lessons"]):
                # previous sections
                previous_subunits = [f"\"{sii + 1}. {s['title']}\"" for sii, s in enumerate(unit["subunits"][:sui])]
                previous_subunits_text = f"We have already covered chapter(s) {', '.join(previous_subunits)} and are now writing a chapter on " if previous_subunits else "We are currently writing the first chapter: "
                # previous lessons
                previous_lessons = [f"{lesson_title}\"" for lii, lesson_title in
                                    enumerate(subunit["lessons"][:li])]
                previous_lessons_text = f"We have already covered the following lessons in the current chapter: {', '.join(previous_lessons)}." if previous_lessons else "You will be writing the first lesson for this chapter."

                section_name = f"{unit['title']} - {subunit['title']}"
                # WIP
                sample = {
                    "unit_title": course["title"].replace("_", " ") + " - " + unit['title'][
                                                                              unit['title'].index(":") + 1:].strip(),
                    "section": section_name,
                    "unit": lesson_title,
                }
                sample["prompt"] = TEMPLATE.substitute(previous_sections=previous_subunits_text,
                                                       previous_sub_units=previous_lessons_text, **sample)
                textbooks.append(sample)

pd.DataFrame(textbooks).to_csv(f"khanacademy_prompts")
