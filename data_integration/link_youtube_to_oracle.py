import os

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
    "": "XDominus EsportsX",  # stays as empty string key
    "VC": "Vici Gaming"
}
if __name__ == "__main__":
    year = 24
    split = "Spring"
    league = "LPL"
    # ________----_____
    folder_path = f"commentary_data/LPL_24_{split.lower()}"
    # Dictionary to store filename: content
    commentary_data = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            name_first_part = filename.split("｜")[0]
            team_vs = name_first_part.split("-")[0]
            game_order = int(name_first_part.split("-")[1][-2])
            with open(file_path, "r", encoding="utf-8") as f:
                commentary_data[filename] = f.read()

    sample_commentary = (team_mapping["AL"], team_mapping["IG"], 1)
    team_a, team_b, game_order = sample_commentary
    mask = (
                   ((match_metadata["team1_name"] == team_a) & (match_metadata["team2_name"] == team_b)) |
                   ((match_metadata["team1_name"] == team_b) & (match_metadata["team2_name"] == team_a))
           ) & (match_metadata["game_order"] == game_order)
    result_row = match_metadata.loc[mask]