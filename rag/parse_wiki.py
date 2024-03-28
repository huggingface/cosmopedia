import json
import lxml.html
from inscriptis import get_text, ParserConfig
from inscriptis.css_profiles import CSS_PROFILES
from multiprocessing import Pool
from smart_open import open
from tqdm import tqdm


def process_file(file_index):
    input_path = f"s3://bigcode-datasets-us-east-1/wikipedia/20240301/enwiki_namespace_0_{file_index}.ndjson"
    output_path = f"s3://bigcode-datasets-us-east-1/wikipedia/20240301/text/enwiki_namespace_0_{file_index}.jsonl"

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in tqdm(
            fin, position=file_index % 64, desc=f"File {file_index}", leave=False
        ):
            data = json.loads(line)
            title = data["name"]
            article_id = data["identifier"]
            html = data["article_body"]["html"]
            root = lxml.html.fromstring(html)

            for cls in [
                "toctitle",
                "infobox",
                "reference",
                "navbox",
                "noprint",
                "metadata",
                "mw-editsection",
            ]:
                xpath = f"//*[contains(@class, '{cls}')]"
                for toremove in root.xpath(xpath):
                    toremove.getparent().remove(toremove)

            ids_to_remove_after = [
                "See_also",
                "References",
                "Notes",
                "Citations",
                "Sources",
                "Further_reading",
                "External_links",
                "Footnotes",
                "Bibliography",
            ]
            for _id in ids_to_remove_after:
                xpath = f"(//h1[./*[@id='{_id}']]|//h1[@id='{_id}']|//h2[./*[@id='{_id}']]|//h2[@id='{_id}']|//h3[./*[@id='{_id}']]|//h3[@id='{_id}'])/self::* | following-sibling::*"
                for toremove in root.xpath(xpath):
                    toremove.getparent().remove(toremove)

            clean_html = lxml.html.tostring(
                root, pretty_print=True, method="html"
            ).decode()
            text = get_text(
                clean_html,
                ParserConfig(display_images=True, css=CSS_PROFILES["strict"]),
            )
            fout.write(
                json.dumps({"id": article_id, "title": title, "text": text}) + "\n"
            )


def main():
    file_indices = range(357)
    with Pool(64) as p:
        p.map(process_file, file_indices)


if __name__ == "__main__":
    main()
