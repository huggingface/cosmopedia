import requests
import json


BASE_URL = "https://en.wikipedia.org/w/api.php"

params = {
    "action": "query",
    "format": "json",
    "list": "allpages",
    "aplimit": "max",
    "apfilterredir": "nonredirects",
}

page_data = []
while True:
    data = requests.get(BASE_URL, params=params).json()

    pages = data["query"]["allpages"]
    for page in pages:
        page_data.append({"id": page["pageid"], "title": page["title"]})

    if "continue" in data:
        params["apcontinue"] = data["continue"]["apcontinue"]
        print(params["apcontinue"])
    else:
        break

with open("wikipedia_titles.jsonl", "w") as f:
    for page in page_data:
        f.write(json.dumps(page) + "\n")
