from matplotlib import pyplot as plt
import numpy as np
from typing import List

def print_distribution(col, bins):

    mean = col.mean()
    std = col.std()

    plt.hist(col, bins=bins)
    plt.axvline(mean, linestyle='--', label=f"Mean = {mean:.3f}", col="red")
    plt.axvline(mean + std, linestyle=':', label=f"+1 Std = {std:.3f}", col="purple")
    plt.axvline(mean - std, linestyle=':', label=f"-1 Std", col="purple")

    plt.legend()
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.title("Histogram with Mean and Std")

    plt.show()

def save_curves(
        epochs: np.ndarray,
        train_losses: List[float],
        val_losses: List[float],
        train_metrics: List[float],
        val_metrics: List[float],
        feature_fn: str,
        target_col: str,
        is_classification: bool = True
) -> None:
    # loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_path = f"loss_curve_{feature_fn}_{target_col}.png"
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"📈 Saved loss curve to {loss_path}")

    # metric
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_metrics, label="Train metric")
    plt.plot(epochs, val_metrics, label="Validation metric")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy" if is_classification else "MAE")
    plt.title("Training vs Validation Metric")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    metric_path = f"metric_curve_{feature_fn}_{target_col}.png"
    plt.savefig(metric_path, dpi=150)
    plt.close()
    print(f"📈 Saved metric curve to {metric_path}")
