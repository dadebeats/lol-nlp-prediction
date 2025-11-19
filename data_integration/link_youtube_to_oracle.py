import os
import pandas as pd
import json
from datetime import datetime
import re
from itertools import product

team_mapping = {
    "NIP": "Ninjas in Pyjamas",
    "WBG": "Weibo Gaming",
    "TES": "Top Esports",
    "BLG": "Bilibili Gaming",
    "JDG": "JD Gaming",
    "LNG": "LNG Esports",
    "AL": "Anyone's Legend",
    "FPX": "FunPlus Phoenix",
    "IG": "Invictus Gaming",
    "LGD": "LGD Gaming",
    "WE": "Team WE",
    "OMG": "Oh My God",
    "UP": "Ultra Prime",
    "TT": "ThunderTalk Gaming",
    "RA": "Rare Atom",
    "EDG": "EDward Gaming",
    "RNG": "Royal Never Give Up",
    "V5": "Victory Five",
    "ES": "eStart",
    "SN": "Suning",
    "RW": "Rogue Warriors",
    "DMO": "Dominus Esports",
    "VC": "Vici Gaming"
}


def filename_to_match_multiindex(filename):
    # Example: "TeamA vs TeamB - Game 2｜something"

    name_first_part = filename.split("｜")[0]
    team_vs = name_first_part.split("-")[0]

    team1 = team_vs.split("vs")[0].strip()
    team2 = team_vs.split("vs")[1].strip()

    # Use regex to find "Game 1", "Game 2", or "Game 3"
    match = re.search(r'\bGame\s*([12345])\b', name_first_part, flags=re.IGNORECASE)
    if match:
        game_order = int(match.group(1))
    else:
        raise ValueError(f"Could not extract game order from filename: {filename}")

    disambiguation_part = filename.split("｜")[1]

    return team1, team2, game_order, disambiguation_part


def get_linked_dataframe(year, league, split):
    folder_path = f"commentary_data/{league}_{year}_{split.lower()}"
    # Dictionary to store filename: content
    commentary_data = {}
    if year == 25:
        split_map = {
            "Winter": "Split 1",
            "Spring": "Split 2",
            "Summer": "Split 3"
        }
        split = split_map[split]

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                commentary_data[filename] = f.read()

    commentary_to_data = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            vid_metadata_path = os.path.join(folder_path, filename.removesuffix(".en.plain.txt")) + ".info.json"
            with open(vid_metadata_path, "r", encoding="utf-8") as f:
                video_metadata = json.load(f)
            team_a, team_b, game_order, disambig = filename_to_match_multiindex(filename)
            team_a = team_mapping[team_a]
            team_b = team_mapping[team_b]
            matches_df = pd.read_csv(f"data_scraping/oracle_elixir_data/processed/{league}_20{year}.csv", index_col=0)
            matches_df = matches_df[matches_df["split"] == split]
            # Convert Oracle date column '2024-01-22 09:24:07' -> datetime.date(2024, 1, 22)
            matches_df["date_only"] = pd.to_datetime(matches_df["date"]).dt.date
            video_date = datetime.strptime(video_metadata["upload_date"], "%Y%m%d").date()

            mask = (
                           ((matches_df["team1_name"] == team_a) & (matches_df["team2_name"] == team_b)) |
                           ((matches_df["team1_name"] == team_b) & (matches_df["team2_name"] == team_a))
                   ) & (matches_df["game_order"] == game_order) & (matches_df["date_only"] == video_date)
            result_row = matches_df.loc[mask]

            if len(result_row) == 1:
                commentary_to_data[filename] = result_row
                print(filename)
            else:
                print("Don't have metadata for: ", team_a, team_b, game_order, disambig)
    print(f"Matches covered by commentary ratio: {len(commentary_to_data)}/{len(matches_df)}")
    output_df = combine_dicts_to_df(commentary_to_data, commentary_data)
    return output_df


def combine_dicts_to_df(dict_df: dict, dict_text: dict) -> pd.DataFrame:
    rows = []
    for key in dict_df.keys():
        # Extract the single-row DataFrame as a dict
        row_data = dict_df[key].iloc[0].to_dict()
        # Add the text from the second dict
        row_data["text"] = dict_text[key]
        # Optionally store the key as a column
        row_data["vid_name"] = key
        rows.append(row_data)
    # Create a DataFrame
    combined_df = pd.DataFrame(rows)
    return combined_df


if __name__ == "__main__":
    years = [21, 22, 23, 24, 25]

    splits = ["Spring", "Summer"]
    league = "LPL"
    # ________----_____
    dfs = []
    for year, split in  list(product(years, splits)) + [(25, "Winter")]:
        dataset_yearly = get_linked_dataframe(year, league, split)
        dfs.append(dataset_yearly)

    dataset = pd.concat(dfs)
    dataset = dataset.set_index("gameid")
    dataset.to_csv("dataset.csv")

