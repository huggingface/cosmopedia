import sys
import json
import time

import requests


# wait until the server is up
while True:
    try:
        requests.post(
            "http://127.0.0.1:9308/search",
            data='{"index": "fineweb", "query": {"match": {"content": "ping"}}}',
        )
        break
    except requests.exceptions.ConnectionError:
        time.sleep(10)
        pass



response = requests.post(
    "http://127.0.0.1:9308/search",
    data=json.dumps({
        "index": "fineweb",
        "query": {
            "match": {
                "content": "Mayonnaise, Baking, Cantonese Cuisine"
            }
        }
    }),
)
print(response.text, file=sys.stderr)
