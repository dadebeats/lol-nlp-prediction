import requests
import os
import json
import pandas as pd
import time

data_list = []
endpoint = "series"
i = 0

while True:
    url = f"https://atlas.abiosgaming.com/v3/{endpoint}?filter=game.id=2&skip={i}"
    # filter=game.id=5
    # Get the secret from environment variable
    client_secret = os.getenv('ABIOS_SECRET')
    headers = {'Abios-Secret': client_secret}
    response = requests.get(url, headers=headers)
    if response.status_code == 429:
        print("Rate limit hit. Sleeping...")
        time.sleep(30)
        continue

    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        break

    data_json = response.json()
    if not data_json:
        break
    data_list.extend(data_json)
    print("Data for endpoint: ", data_json)
    i += 50

with open(f"data_scraping/abios_data/{endpoint}.json", "w") as f:
    json.dump(data_list, f, indent=True)

df = pd.DataFrame(data_list)





