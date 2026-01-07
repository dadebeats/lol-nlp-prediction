from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Hashable, List, Sequence, Union
from collections import Counter

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Patch, Circle

def find_pca_outliers(df_pca: pd.DataFrame, x_col="pca_x", y_col="pca_y", top_n=5) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Finds data points which lie the furthest from the centroid in the PCA space. Assumes PCA already attached to df.
    :param df_pca:
    :param x_col:
    :param y_col:
    :param top_n:
    :return:
    """
    # Extract PCA coordinates
    X = df_pca[[x_col, y_col]].to_numpy()

    # Compute PCA centroid
    centroid = X.mean(axis=0)

    # Compute Euclidean distance from centroid
    distances = np.linalg.norm(X - centroid, axis=1)

    # Store distances in a new column
    df = df_pca.copy()
    df["pca_distance_from_mean"] = distances

    # Sort descending by distance → farthest points first
    outliers = df.sort_values("pca_distance_from_mean", ascending=False).head(top_n)

    return outliers, centroid

# ------------------------
# Data helpers
# ------------------------
def load_parquet(path: str) -> pd.DataFrame:
    """Load a parquet dataset (no sampling here; do it outside if needed)."""
    return pd.read_parquet(path)


def stack_embeddings(df: pd.DataFrame, emb_col: str = "embedding") -> np.ndarray:
    """Stack a DataFrame column of vectors/lists into a 2D numpy array (N, D)."""
    return np.vstack(df[emb_col].to_numpy())


# ------------------------
# PCA utilities
# ------------------------
def compute_pca_matrix(
    X: np.ndarray,
    n_components: int = 2,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, PCA]:
    """Run PCA on X and return (X_pca, explained_variance_ratio, pca_obj)."""
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_
    return X_pca, evr, pca


def attach_pca_columns(
    df: pd.DataFrame,
    X_pca: np.ndarray,
    x_col: str = "pca_x",
    y_col: str = "pca_y"
) -> pd.DataFrame:
    """Attach first two PCA columns to a copy of df."""
    out = df.copy()
    out[x_col] = X_pca[:, 0]
    out[y_col] = X_pca[:, 1]
    return out


def run_pca_and_attach(
    df: pd.DataFrame,
    *,
    emb_col: str = "embedding",
    n_components: int = 2,
    random_state: Optional[int] = 42,
    x_col: str = "pca_x",
    y_col: str = "pca_y",
    n_outliers_to_omit: int = 50,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, PCA]:
    """
    Convenience: run PCA on the *full* df[emb_col], attach pca_x/pca_y,
    and return (df_pca, X_pca, explained_variance_ratio, pca_obj).
    """
    X = stack_embeddings(df, emb_col=emb_col)
    X_pca, evr, pca_obj = compute_pca_matrix(X, n_components=n_components, random_state=random_state)
    print(f"Explained variance ratio (PC1, PC2.. PCN): {evr}")
    print(f"Total explained variance by first {n_components} PCs: {evr[:n_components].sum() * 100:.2f}%")

    df_pca = attach_pca_columns(df, X_pca, x_col=x_col, y_col=y_col)
    outliers, centroid = find_pca_outliers(df_pca, top_n=len(df_pca))
    return outliers.tail(len(df_pca) - n_outliers_to_omit), X_pca, evr, pca_obj


# ------------------------
# Generic key helpers (support 1+ columns)
# ------------------------
def row_key(row: pd.Series, cols: Union[str, Sequence[str]]) -> Hashable:
    if isinstance(cols, str):
        return row[cols]
    return tuple(row[c] for c in cols)


def unique_keys(df: pd.DataFrame, cols: Union[str, Sequence[str]]) -> List[Hashable]:
    if isinstance(cols, str):
        return list(pd.unique(df[cols]))
    return list(dict.fromkeys(tuple(v) for v in df[list(cols)].itertuples(index=False, name=None)))


def most_frequent_keys(
    df: pd.DataFrame,
    cols: Union[str, Sequence[str]],
    top_n: int = 8
) -> set:
    if isinstance(cols, str):
        counts = Counter(df[cols].tolist())
        return {k for k, _ in counts.most_common(top_n)}
    tuples = [tuple(v) for v in df[list(cols)].itertuples(index=False, name=None)]
    counts = Counter(tuples)
    return {k for k, _ in counts.most_common(top_n)}


def key_to_label(key: Hashable, sep: str = " ") -> str:
    if isinstance(key, tuple):
        return sep.join("" if v is None else str(v) for v in key)
    return "" if key is None else str(key)


# ------------------------
# Color mapping (categorical)
# ------------------------
def build_color_map_for_keys(
    keys: List[Hashable],
    cmap_name: str = "tab20"
) -> Dict[Hashable, Tuple[float, float, float, float]]:
    cmap = plt.cm.get_cmap(cmap_name, len(keys) if len(keys) > 0 else 1)
    return {k: cmap(i) for i, k in enumerate(keys)}


def build_color_map_for_cols(
    df: pd.DataFrame,
    cols: Union[str, Sequence[str]],
    cmap_name: str = "tab20"
) -> Dict[Hashable, Tuple[float, float, float, float]]:
    keys = unique_keys(df, cols)
    return build_color_map_for_keys(keys, cmap_name=cmap_name)


# ------------------------
# Plotting
# ------------------------
def _auto_radius(df: pd.DataFrame, x_col: str, y_col: str, radius_factor: float) -> float:
    xr = float(df[x_col].max() - df[x_col].min())
    yr = float(df[y_col].max() - df[y_col].min())
    return radius_factor * max(xr, yr)


def plot_half_markers_by_two_cols(
    df: pd.DataFrame,
    left_col: str,
    right_col: str,
    *,
    color_map: Optional[Dict[Hashable, Tuple[float, float, float, float]]] = None,
    cmap_name: str = "tab20",
    x_col: str = "pca_x",
    y_col: str = "pca_y",
    legend_top_n: int = 8,
    radius_factor: float = 0.0125,
    figsize: Tuple[int, int] = (9, 7),
    title: str = None,
    save_path: Optional[str] = None,
    show: bool = True,
    close: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Two-half markers (left/right) colored by values in `left_col` and `right_col`.
    Expects df to already have pca_x/pca_y.
    """
    if color_map is None:
        keys = unique_keys(pd.DataFrame({
            "__u__": pd.concat([df[left_col], df[right_col]], ignore_index=True)
        }), "__u__")
        color_map = build_color_map_for_keys(keys, cmap_name=cmap_name)

    fig, ax = plt.subplots(figsize=figsize)
    r = _auto_radius(df, x_col=x_col, y_col=y_col, radius_factor=radius_factor)

    for _, row in df.iterrows():
        x, y = float(row[x_col]), float(row[y_col])
        c_left = color_map.get(row[left_col], (0.5, 0.5, 0.5, 1.0))
        c_right = color_map.get(row[right_col], (0.7, 0.7, 0.7, 1.0))

        w1 = Wedge(center=(x, y), r=r, theta1=90, theta2=270,
                   facecolor=c_left, edgecolor="k", linewidth=0.3, alpha=0.9)
        w2 = Wedge(center=(x, y), r=r, theta1=270, theta2=90,
                   facecolor=c_right, edgecolor="k", linewidth=0.3, alpha=0.9)
        ax.add_patch(w1)
        ax.add_patch(w2)

    ax.set_title(title or f"PCA — Half markers by [{left_col}] vs [{right_col}]")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    ax.set_xlim(df[x_col].min() - r * 2, df[x_col].max() + r * 2)
    ax.set_ylim(df[y_col].min() - r * 2, df[y_col].max() + r * 2)

    # Legend (top-N across both columns)
    top_set = most_frequent_keys(
        pd.DataFrame({
            "__u__": pd.concat([df[left_col], df[right_col]], ignore_index=True)
        }),
        "__u__",
        top_n=legend_top_n
    )
    legend_handles = [Patch(color=color_map[k], label=key_to_label(k)) for k in color_map if k in top_set]
    if legend_handles:
        leg = ax.legend(handles=legend_handles, title=f"Top {legend_top_n}",
                        bbox_to_anchor=(1.02, 1), loc="upper left")
        for lh in leg.legend_handles:
            lh.set_linewidth(0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return fig, ax


def plot_full_markers_by_cols(
    df: pd.DataFrame,
    cols: Union[str, Sequence[str]],
    *,
    # categorical options
    color_map: Optional[Dict[Hashable, Tuple[float, float, float, float]]] = None,
    cmap_name: str = "tab20",
    # shared options
    x_col: str = "pca_x",
    y_col: str = "pca_y",
    legend_top_n: int = 8,
    radius_factor: float = 0.0125,
    figsize: Tuple[int, int] = (9, 7),
    title: str = None,
    save_path: Optional[str] = None,
    show: bool = True,
    close: bool = False,
    use_circle: bool = True,
    # continuous gradient options
    continuous: bool = False,
    continuous_cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_colorbar: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Full markers (single color) by one or multiple columns.

    - Categorical mode (continuous=False):
        * If `cols` is a sequence, each point's key is a tuple (e.g., (year, split)).
        * Colors are assigned per unique key.
    - Continuous mode (continuous=True):
        * `cols` must be a single column name (str).
        * Values are mapped with a gradient colormap (continuous_cmap).
    Expects df to already have pca_x/pca_y.
    """
    fig, ax = plt.subplots(figsize=figsize)
    r = _auto_radius(df, x_col=x_col, y_col=y_col, radius_factor=radius_factor)

    norm = None
    scalar_cmap = None

    if continuous:
        if not isinstance(cols, str):
            raise ValueError("continuous=True requires `cols` to be a single column name (str).")

        values = pd.to_numeric(df[cols], errors="coerce")
        if values.notna().any():
            vmin = float(values.min()) if vmin is None else vmin
            vmax = float(values.max()) if vmax is None else vmax
        else:
            vmin = 0.0 if vmin is None else vmin
            vmax = 1.0 if vmax is None else vmax

        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        scalar_cmap = plt.cm.get_cmap(continuous_cmap)
    else:
        if color_map is None:
            color_map = build_color_map_for_cols(df, cols, cmap_name=cmap_name)

    for _, row in df.iterrows():
        x, y = float(row[x_col]), float(row[y_col])

        if continuous:
            val = pd.to_numeric(pd.Series([row[cols]]), errors="coerce").iloc[0]
            if pd.isna(val):
                c = (0.8, 0.8, 0.8, 0.4)
            else:
                c = scalar_cmap(norm(val))
        else:
            key = row_key(row, cols)
            c = color_map.get(key, (0.6, 0.6, 0.6, 1.0))

        if use_circle:
            circ = Circle(xy=(x, y), radius=r, facecolor=c, edgecolor="k", linewidth=0.3, alpha=0.9)
            ax.add_patch(circ)
        else:
            w = Wedge(center=(x, y), r=r, theta1=0, theta2=360,
                      facecolor=c, edgecolor="k", linewidth=0.3, alpha=0.9)
            ax.add_patch(w)

    if title is None:
        if continuous:
            title = f"PCA — Colored by continuous [{cols}]"
        else:
            if isinstance(cols, str):
                title = f"PCA — Colored by [{cols}]"
            else:
                title = f"PCA — Colored by {list(cols)}"

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    ax.set_xlim(df[x_col].min() - r * 2, df[x_col].max() + r * 2)
    ax.set_ylim(df[y_col].min() - r * 2, df[y_col].max() + r * 2)

    if continuous:
        if show_colorbar and norm is not None and scalar_cmap is not None:
            sm = plt.cm.ScalarMappable(norm=norm, cmap=scalar_cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label(cols)
    else:
        top_set = most_frequent_keys(df, cols, top_n=legend_top_n)
        legend_handles = []
        for k, c in color_map.items():
            if k in top_set:
                legend_handles.append(Patch(color=c, label=key_to_label(k)))
        if legend_handles:
            leg = ax.legend(handles=legend_handles, title=f"Top {legend_top_n}",
                            bbox_to_anchor=(1.02, 1), loc="upper left")
            for lh in leg.legend_handles:
                lh.set_linewidth(0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return fig, ax


# ------------------------
# High-level "pipelines"
# ------------------------
def pca_half_markers_pipeline(
    df_pca: pd.DataFrame,
    *,
    left_col: str = "team1_name",
    right_col: str = "team2_name",
    legend_top_n: int = 8,
    radius_factor: float = 0.0125,
    cmap_name: str = "tab20",
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Pipeline for half markers by two columns (e.g., team vs team).
    Expects df_pca to already contain PCA columns (pca_x, pca_y).
    """
    # shared color map from union of left/right columns
    all_keys_df = pd.DataFrame({"__u__": pd.concat([df_pca[left_col], df_pca[right_col]], ignore_index=True)})
    cmap = build_color_map_for_cols(all_keys_df, "__u__", cmap_name=cmap_name)

    fig, ax = plot_half_markers_by_two_cols(
        df_pca, left_col, right_col,
        color_map=cmap,
        cmap_name=cmap_name,
        legend_top_n=legend_top_n,
        radius_factor=radius_factor,
        title=f"PCA — Half markers by [{left_col}] vs [{right_col}]",
        save_path=save_path,
        show=show,
        close=not show,
    )
    return fig, ax


def pca_full_markers_pipeline(
    df_pca: pd.DataFrame,
    *,
    color_cols: Union[str, Sequence[str]] = ("year", "split"),
    legend_top_n: int = 8,
    radius_factor: float = 0.0125,
    cmap_name: str = "tab20",
    save_path: Optional[str] = None,
    show: bool = False,
    continuous: bool = False,
    continuous_cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_colorbar: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Pipeline for full markers colored by one or multiple columns.

    - Categorical: color_cols can be str or sequence, continuous=False.
    - Continuous: color_cols must be a single column name (str), continuous=True.
    Expects df_pca to already contain PCA columns.
    """
    title = (
        f"PCA — Colored by continuous [{color_cols}]"
        if continuous and isinstance(color_cols, str)
        else f"PCA — Colored by {list(color_cols) if isinstance(color_cols, (list, tuple)) else color_cols}"
    )

    cmap = None
    if not continuous:
        cmap = build_color_map_for_cols(df_pca, color_cols, cmap_name=cmap_name)

    fig, ax = plot_full_markers_by_cols(
        df_pca,
        color_cols,
        color_map=cmap,
        cmap_name=cmap_name,
        legend_top_n=legend_top_n,
        radius_factor=radius_factor,
        title=title,
        save_path=save_path,
        show=show,
        close=not show,
        use_circle=True,
        continuous=continuous,
        continuous_cmap=continuous_cmap,
        vmin=vmin,
        vmax=vmax,
        show_colorbar=show_colorbar,
    )
    return fig, ax

if __name__ == "__main__":
    # Load the embedded dataset (must contain an 'embedding' column and match metadata).
    df = load_parquet("m-player+team_asr-corr_mdl-openai-oai_emb3_ck-t512-o256.parquet")

    # Quick derived features for optional coloring/filtering in PCA plots.
    df["total_kills"] = df["team1_kills"] + df["team2_kills"]
    df["kill_diff_abs"] = (df["team1_kills"] - df["team2_kills"]).abs()

    # Optional: restrict to years included in the analysis (keeps PCA space consistent).
    # ~ Pretend the year 2025 does not exist in our data.
    df = df[df["year"] < 2025].copy()

    # Run PCA ONCE on the full dataset, then do any filtering on the resulting df_pca.
    n_components = 2
    df_pca, X_pca, evr, pca = run_pca_and_attach(
        df,
        emb_col="embedding",
        n_components=n_components,
        random_state=42,
        x_col="pca_x",
        y_col="pca_y",
        n_outliers_to_omit=300,  # drop farthest points to improve plot readability
    )

    # Explained variance diagnostics (how much of the embedding variance PCA captures).
    print(f"Explained variance ratio: {evr}")
    print(f"Total explained variance (first {n_components} PCs): {evr[:n_components].sum() * 100:.2f}%")

    # --- Plot A: team vs team half-markers (left = team1, right = team2) ---
    pca_half_markers_pipeline(
        df_pca,
        left_col="team1_name",
        right_col="team2_name",
        legend_top_n=16,
        radius_factor=0.0125,
        cmap_name="tab20",
        save_path="pca_half_pies_by_teams.png",
        show=True,
    )
    print("Saved: pca_half_pies_by_teams.png")

    # --- Plot B1: categorical coloring (example: year) ---
    pca_full_markers_pipeline(
        df_pca,
        color_cols="year",
        legend_top_n=10,
        radius_factor=0.0125,
        cmap_name="tab20",
        save_path="pca_by_year_categorical.png",
        show=True,
        continuous=False,
    )
    print("Saved: pca_by_year_categorical.png")

    # --- Plot B2: continuous coloring (example: absolute kill difference) ---
    # Filter here affects only which points are shown, not the PCA itself.
    df_kd = df_pca[(df_pca["kill_diff_abs"] > 16) | (df_pca["kill_diff_abs"] < 6)]
    pca_full_markers_pipeline(
        df_kd,
        color_cols="kill_diff_abs",
        legend_top_n=10,
        radius_factor=0.0125,
        continuous=True,
        continuous_cmap="viridis",
        save_path="pca_by_kill_diff_abs_continuous.png",
        show=True,
    )
    print("Saved: pca_by_kill_diff_abs_continuous.png")

    # --- Plot B3: continuous coloring (example: game length) ---
    df_len = df_pca[(df_pca["gamelength"] > 2200) | (df_pca["gamelength"] < 1600)]
    pca_full_markers_pipeline(
        df_len,
        color_cols="gamelength",
        legend_top_n=10,
        radius_factor=0.0125,
        continuous=True,
        continuous_cmap="viridis",
        save_path="pca_by_gamelength_continuous.png",
        show=True,
    )
    print("Saved: pca_by_gamelength_continuous.png")

    # --- Inspect PCA outliers (farthest points from the PCA centroid) ---
    outliers, centroid = find_pca_outliers(df_pca, top_n=5)
    print("PCA centroid:", centroid)
    print("\nTop 5 farthest rows:")
    print(outliers[["pca_x", "pca_y", "pca_distance_from_mean", "text"]])

