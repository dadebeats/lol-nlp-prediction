import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def apply_pca_to_embeddings(
        df: pd.DataFrame,
        emb_col: str = "embedding",
        n_components: int = 0,
) -> pd.DataFrame:
    """
    If n_components > 0, fit PCA on the entire df[emb_col] (assumed vector-like)
    and overwrite that column with reduced-dim embeddings.

    If n_components <= 0, returns df unchanged.
    """
    if n_components is None or n_components <= 0:
        return df

    # Stack embeddings into (N, D)
    emb_mat = np.vstack(df[emb_col].to_numpy())  # each cell is list/np.array

    pca = PCA(n_components=n_components, random_state=42)
    emb_reduced = pca.fit_transform(emb_mat).astype(np.float32)  # (N, n_components)

    # Write back: each row gets a 1D np.array of length n_components
    df = df.copy()
    df[emb_col] = list(emb_reduced)

    print(
        f"🔎 PCA applied on '{emb_col}': original_dim={emb_mat.shape[1]}, "
        f"new_dim={n_components}, explained_var={pca.explained_variance_ratio_.sum() * 100:.2f}%"
    )
    return df