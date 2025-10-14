import pandas as pd


def extract_match_metadata(df):
    # Filter to one match worth of rows
    matches = []
    for gameid, match_df in df.groupby("gameid"):
        # Ignore incomplete matches if you want
        # match_df = match_df[match_df['datacompleteness'] == 'complete']
        # Basic metadata (shared across all rows)
        meta = {
            "gameid": gameid,
            "date": match_df["date"].iloc[0],
            "league": match_df["league"].iloc[0],
            "year": match_df["year"].iloc[0],
            "split": match_df["split"].iloc[0],
            "patch": match_df["patch"].iloc[0],
            "game_order": match_df["game"].iloc[0],
        }
        # Split by side
        position_order = ["top", "jng", "mid", "bot", "sup"]
        teams = match_df[match_df["position"] == "team"]  # team rows have no position
        players = match_df[match_df["position"].isin(position_order)]  # player rows have positions
        # Sort players within each side by position order
        sides = teams["side"].unique()

        # Extract teams
        team1_side = sides[0]
        team2_side = sides[1]
        team1 = teams.loc[teams["side"] == team1_side].iloc[0]
        team2 = teams.loc[teams["side"] == team2_side].iloc[0]
        meta.update({
            "team1_name": team1["teamname"],
            "team2_name": team2["teamname"],
            "team1_side": team1_side,
            "team2_side": team2_side,
            "team1_bans": [team1[f"ban{i}"] for i in range(1, 6) if pd.notna(team1[f"ban{i}"])],
            "team2_bans": [team2[f"ban{i}"] for i in range(1, 6) if pd.notna(team2[f"ban{i}"])],
            "team1_result": team1["result"],

            #"team1_kills": team1["kills"],

        })
        # Extract player & champion names by team
        team1_players = players.loc[players["side"] == team1_side]["playername"].tolist()
        team2_players = players.loc[players["side"] == team2_side]["playername"].tolist()
        team1_champions = players.loc[players["side"] == team1_side]["champion"].tolist()
        team2_champions = players.loc[players["side"] == team2_side]["champion"].tolist()
        meta.update({
            "team1_players": team1_players,
            "team2_players": team2_players,
            "team1_champions": team1_champions,
            "team2_champions": team2_champions
        })
        matches.append(meta)
    return pd.DataFrame(matches)


if __name__ == '__main__':
    for year in range(20, 26):

        #split = "Spring"
        league = "LPL"
        # ________----_____
        raw_data = pd.read_csv(f"data_scraping/oracle_elixir_data/20{year}_LoL_esports_match_data_from_OraclesElixir.csv")
        filtered_data = raw_data[(raw_data.league == league)]
        matches_metadata = extract_match_metadata(filtered_data)
        matches_metadata.to_csv(f"data_scraping/oracle_elixir_data/processed/{league}_20{year}.csv")

