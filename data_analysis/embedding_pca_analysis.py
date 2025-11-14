# pca_halfpies.py

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Iterable, Tuple, Optional, Hashable, List, Sequence, Union
from collections import Counter

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Patch, Circle


# ------------------------
# Data helpers
# ------------------------
def load_parquet(path: str, sample_n: Optional[int] = None, random_state: int = 42) -> pd.DataFrame:
    """Load a parquet dataset, optionally sampling n rows."""
    df = pd.read_parquet(path)
    if sample_n is not None and sample_n > 0 and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=random_state)
    return df


def stack_embeddings(df: pd.DataFrame, emb_col: str = "embedding") -> np.ndarray:
    """Stack a DataFrame column of vectors/lists into a 2D numpy array (N, D)."""
    return np.vstack(df[emb_col].to_numpy())


# ------------------------
# PCA utilities
# ------------------------
def compute_pca_matrix(
    X: np.ndarray,
    n_components: int = 2,
    random_state: int = 42
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


# ------------------------
# Generic key helpers (support 1+ columns)
# ------------------------
def row_key(row: pd.Series, cols: Union[str, Sequence[str]]) -> Hashable:
    """
    Build a hashable key from row by `cols`.
    - If cols is a str -> returns the scalar value
    - If cols is a sequence -> returns a tuple of values (preserves None)
    """
    if isinstance(cols, str):
        return row[cols]
    return tuple(row[c] for c in cols)


def unique_keys(df: pd.DataFrame, cols: Union[str, Sequence[str]]) -> List[Hashable]:
    """Collect unique category keys from one or multiple columns."""
    if isinstance(cols, str):
        return list(pd.unique(df[cols]))
    # composite: create tuple keys
    return list(dict.fromkeys(tuple(v) for v in df[list(cols)].itertuples(index=False, name=None)))


def most_frequent_keys(
    df: pd.DataFrame,
    cols: Union[str, Sequence[str]],
    top_n: int = 8
) -> set:
    """
    Return the top-N most frequent keys across the specified cols.
    For composite cols, counts by tuple.
    """
    if isinstance(cols, str):
        counts = Counter(df[cols].tolist())
        return {k for k, _ in counts.most_common(top_n)}
    # composite: count tuples
    tuples = [tuple(v) for v in df[list(cols)].itertuples(index=False, name=None)]
    counts = Counter(tuples)
    return {k for k, _ in counts.most_common(top_n)}


def key_to_label(key: Hashable, sep: str = " ") -> str:
    """Nicely render a key (tuple -> 'a b', scalar -> str)."""
    if isinstance(key, tuple):
        return sep.join("" if v is None else str(v) for v in key)
    return "" if key is None else str(key)


# ------------------------
# Color mapping (generic)
# ------------------------
def build_color_map_for_keys(
    keys: List[Hashable],
    cmap_name: str = "tab20"
) -> Dict[Hashable, Tuple[float, float, float, float]]:
    """
    Build a key -> RGBA map for arbitrary (possibly tuple) keys.
    """
    cmap = plt.cm.get_cmap(cmap_name, len(keys) if len(keys) > 0 else 1)
    return {k: cmap(i) for i, k in enumerate(keys)}


def build_color_map_for_cols(
    df: pd.DataFrame,
    cols: Union[str, Sequence[str]],
    cmap_name: str = "tab20"
) -> Dict[Hashable, Tuple[float, float, float, float]]:
    """Build color map for all unique keys derived from `cols`."""
    keys = unique_keys(df, cols)
    return build_color_map_for_keys(keys, cmap_name=cmap_name)


# ------------------------
# Plotting
# ------------------------
def _auto_radius(df: pd.DataFrame, x_col: str, y_col: str, radius_factor: float) -> float:
    """Choose a radius in data units based on PCA spread and a factor."""
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
    """
    if color_map is None:
        # colors come from the union of both columns
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

        # left half 90–270, right half 270–90
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
    use_circle: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Full markers (single color) by one or multiple columns.
    - If `cols` is a sequence, each point's key is a tuple (e.g., (year, split)).
    - Colors are assigned per unique key.
    """
    if color_map is None:
        color_map = build_color_map_for_cols(df, cols, cmap_name=cmap_name)

    fig, ax = plt.subplots(figsize=figsize)
    r = _auto_radius(df, x_col=x_col, y_col=y_col, radius_factor=radius_factor)

    for _, row in df.iterrows():
        x, y = float(row[x_col]), float(row[y_col])
        key = row_key(row, cols)
        c = color_map.get(key, (0.6, 0.6, 0.6, 1.0))

        if use_circle:
            circ = Circle(xy=(x, y), radius=r, facecolor=c, edgecolor="k", linewidth=0.3, alpha=0.9)
            ax.add_patch(circ)
        else:
            # Wedge 0–360 is visually identical to a circle
            w = Wedge(center=(x, y), r=r, theta1=0, theta2=360,
                      facecolor=c, edgecolor="k", linewidth=0.3, alpha=0.9)
            ax.add_patch(w)

    if title is None:
        if isinstance(cols, str):
            title = f"PCA — Colored by [{cols}]"
        else:
            title = f"PCA — Colored by {list(cols)}"
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    ax.set_xlim(df[x_col].min() - r * 2, df[x_col].max() + r * 2)
    ax.set_ylim(df[y_col].min() - r * 2, df[y_col].max() + r * 2)

    # Legend (top-N by specified columns)
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
# High-level pipelines
# ------------------------
def pca_half_markers_pipeline(
    parquet_path: str = "dataset.parquet",
    *,
    emb_col: str = "embedding",
    left_col: str = "team1_name",
    right_col: str = "team2_name",
    sample_n: Optional[int] = 200,
    random_state: int = 42,
    n_components: int = 2,
    legend_top_n: int = 8,
    radius_factor: float = 0.0125,
    cmap_name: str = "tab20",
    save_path: str = "pca_half_markers.png",
    show: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Pipeline for half markers by two columns (e.g., team vs team).
    """
    df = load_parquet(parquet_path, sample_n=sample_n, random_state=random_state)
    X = stack_embeddings(df, emb_col=emb_col)
    X_pca, evr, _ = compute_pca_matrix(X, n_components=n_components, random_state=random_state)
    print(f"Explained variance ratio (PC1, PC2): {evr}")
    print(f"Total explained variance by first 2 PCs: {evr.sum() * 100:.2f}%")

    df_pca = attach_pca_columns(df, X_pca, x_col="pca_x", y_col="pca_y")

    # shared color map from union of left/right columns
    all_keys_df = pd.DataFrame({"__u__": pd.concat([df_pca[left_col], df_pca[right_col]], ignore_index=True)})
    cmap = build_color_map_for_cols(all_keys_df, "__u__", cmap_name=cmap_name)

    plot_half_markers_by_two_cols(
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
    return df_pca, X_pca, evr


def pca_full_markers_pipeline(
    parquet_path: str = "dataset.parquet",
    *,
    emb_col: str = "embedding",
    color_cols: Union[str, Sequence[str]] = ("year", "split"),
    sample_n: Optional[int] = 200,
    random_state: int = 42,
    n_components: int = 2,
    legend_top_n: int = 8,
    radius_factor: float = 0.0125,
    cmap_name: str = "tab20",
    save_path: str = "pca_full_markers.png",
    show: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Pipeline for full markers colored by one or multiple columns.
    Example: color_cols=("year","split") to represent season in a single color.
    """
    df = load_parquet(parquet_path, sample_n=sample_n, random_state=random_state)
    X = stack_embeddings(df, emb_col=emb_col)
    X_pca, evr, _ = compute_pca_matrix(X, n_components=n_components, random_state=random_state)
    print(f"Explained variance ratio (PC1, PC2): {evr}")
    print(f"Total explained variance by first 2 PCs: {evr.sum() * 100:.2f}%")

    df_pca = attach_pca_columns(df, X_pca, x_col="pca_x", y_col="pca_y")
    cmap = build_color_map_for_cols(df_pca, color_cols, cmap_name=cmap_name)

    plot_full_markers_by_cols(
        df_pca,
        color_cols,
        color_map=cmap,
        cmap_name=cmap_name,
        legend_top_n=legend_top_n,
        radius_factor=radius_factor,
        title=f"PCA — Colored by {list(color_cols) if isinstance(color_cols, (list, tuple)) else color_cols}",
        save_path=save_path,
        show=show,
        close=not show,
        use_circle=True,
    )
    return df_pca, X_pca, evr


# ------------------------
# Script entry point (short & readable)
# ------------------------
if __name__ == "__main__":
    # (A) Half markers by two columns — keeps old behavior (team vs team)
    pca_half_markers_pipeline(
        parquet_path="dataset.parquet",
        emb_col="embedding_masked",
        left_col="team1_name",
        right_col="team2_name",
        sample_n=400,
        random_state=42,
        n_components=2,
        legend_top_n=8,
        radius_factor=0.0125,
        cmap_name="tab20",
        save_path="pca_half_pies_by_teams.png",
        show=False,
    )
    print("Saved: pca_half_pies_by_teams.png")

    # (B) Full markers by composite columns — e.g., season coloring by (year, split)
    # Uncomment to run:
    pca_full_markers_pipeline(
        parquet_path="dataset.parquet",
        emb_col="embedding_masked",
        color_cols=("patch"),   # try also: "patch"
        sample_n=400,
        random_state=42,
        n_components=2,
        legend_top_n=10,
        radius_factor=0.0125,
        cmap_name="tab20",
        save_path="pca_by_year_split.png",
        show=False,
    )
    print("Saved: pca_by_year_split.png")
