import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from matplotlib.patches import Wedge, Patch


if __name__ == "__main__":
    # --- Load your data (if not already loaded)
    df = pd.read_parquet("dataset.parquet")
    df = df.sample(n=200)
    # --- Stack all embeddings into a matrix
    X = np.vstack(df["embedding"].to_numpy())  # shape (N, D)

    # --- Run PCA to 2D
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_
    print(f"Explained variance ratio (PC1, PC2): {evr}")
    print(f"Total explained variance by first 2 PCs: {evr.sum()*100:.2f}%")

    # --- Add results to df for convenience
    df["pca_x"] = X_pca[:, 0]
    df["pca_y"] = X_pca[:, 1]

    # --- Build team→color map
    teams = pd.unique(pd.concat([df["team1_name"], df["team2_name"]], ignore_index=True))
    cmap = plt.cm.get_cmap("tab20", len(teams))
    team2color = {team: cmap(i) for i, team in enumerate(teams)}

    # Optional: limit legend to top-N most frequent teams (keeps legend readable)
    N_LEGEND = 8
    counts = Counter(pd.concat([df["team1_name"], df["team2_name"]], ignore_index=True))
    top_teams = {t for t,_ in counts.most_common(N_LEGEND)}

    fig, ax = plt.subplots(figsize=(9,7))

    # Choose a radius in data units (scales with your PCA spread)
    xr = df["pca_x"].max() - df["pca_x"].min()
    yr = df["pca_y"].max() - df["pca_y"].min()
    r = 0.0125 * max(xr, yr)  # tweak if you want bigger/smaller markers

    for _, row in df.iterrows():
        x, y = float(row["pca_x"]), float(row["pca_y"])
        c1 = team2color[row["team1_name"]]
        c2 = team2color[row["team2_name"]]

        # Two half-circles (0–180°, 180–360°). Flip if you prefer left/right vs top/bottom.
        w1 = Wedge(center=(x, y), r=r, theta1=90, theta2=270, facecolor=c1, edgecolor="k", linewidth=0.3, alpha=0.9)
        w2 = Wedge(center=(x, y), r=r, theta1=270, theta2=90, facecolor=c2, edgecolor="k", linewidth=0.3, alpha=0.9)

        ax.add_patch(w1)
        ax.add_patch(w2)

    ax.set_title("PCA of Match Embeddings — Half-and-Half Team Colors")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # Make axes limits snug around data with a small margin
    ax.set_xlim(df["pca_x"].min() - r*2, df["pca_x"].max() + r*2)
    ax.set_ylim(df["pca_y"].min() - r*2, df["pca_y"].max() + r*2)

    # Legend (top-N)
    legend_handles = [Patch(color=team2color[t], label=t) for t in teams if t in top_teams]
    leg = ax.legend(handles=legend_handles, title="Teams (top N)", bbox_to_anchor=(1.02, 1), loc="upper left")
    for lh in leg.legend_handles:
        lh.set_linewidth(0)  # no borders in legend

    plt.tight_layout()
    plt.savefig("pca_half_pies_by_teams.png", dpi=300)
    plt.close()
    print("Saved: pca_half_pies_by_teams.png")