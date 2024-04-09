import requests
import json
from lxml import html
import time
from tqdm import tqdm


cat_data = []

top_html = requests.get(
    "https://www.bisg.org/complete-bisac-subject-headings-list"
).content
top_tree = html.fromstring(top_html)
top_links = top_tree.xpath(
    '//*[@id="content"]/div[1]/div[2]/div[2]/novi-content-wrapper/table/tbody/tr/td/ul/li/a'
)
for top_link in tqdm(top_links):
    time.sleep(2)
    cat_html = requests.get(top_link.attrib["href"]).content
    cat_tree = html.fromstring(cat_html)
    categories = cat_tree.xpath(
        '//p[re:test(normalize-space(), "^[A-Z]+[0-9]+") and b]',
        namespaces={"re": "http://exslt.org/regular-expressions"},
    )

    for category in categories:
        code = category.xpath("text()")[0].strip()
        top_cat = category.xpath("b/text()")[0].strip()
        subcat = "".join(category.itertext()).split("/", 1)[1].strip()
        subcat = subcat.split("(see ", 1)[0].strip(" *")

        cat_data.append({"code": code, "top_category": top_cat, "subcategory": subcat})

with open("bisac.json", "w") as f:
    json.dump(cat_data, f, indent=2)
