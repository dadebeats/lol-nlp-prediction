import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Dict, List
import torch.optim as optim
import numpy as np
import optuna
from optuna.trial import Trial

# --- helper: train loop (shared by optuna + final runs) ---

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[optim.Optimizer],
    device: torch.device,
    is_classification: bool
) -> Tuple[float, float]:
    """
    Execute a single training or evaluation epoch.

    If an optimizer is provided, the model is trained and parameters are updated (train data).
    Otherwise, the model is evaluated without gradient updates (val data).

    Returns the average loss and the primary evaluation metric:
    accuracy for classification tasks or mean absolute error for regression.
    """
    train = optimizer is not None
    model.train(train)

    total_loss = 0.0
    metric_sum = 0.0
    n = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = criterion(logits, yb)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        bs = xb.size(0)
        total_loss += loss.item() * bs
        n += bs

        if is_classification:
            preds = (logits.sigmoid() >= 0.5).long()
            metric_sum += (preds == yb.long()).sum().item()  # correct
        else:
            metric_sum += (logits - yb).abs().sum().item()  # MAE sum

    avg_loss = total_loss / max(n, 1)
    avg_metric = metric_sum / max(n, 1)  # accuracy or MAE
    return avg_loss, avg_metric

def train_eval(
    *,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    is_classification: bool,
    epochs: int,
    save_path: Optional[str] = None,
    trial: Optional[Trial] = None,
    objective_mode: str = "acc",   # "acc" or "loss" for classification; regression uses MAE
) -> Tuple[float, Dict[str, List[float]]]:
    """
    Run a full training loop consisting of multiple epochs with interleaved
    training and validation phases.

    Tracks loss and task-specific metrics over time, applies learning rate
    scheduling based on validation performance, and optionally supports
    Optuna-based pruning and checkpointing.

    Returns the best validation score achieved and a dictionary containing
    training and validation curves.
    """
    best = -np.inf
    curves = {"train_loss": [], "val_loss": [], "train_metric": [], "val_metric": []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_metric = run_epoch(
            model=model, loader=train_loader, criterion=criterion, optimizer=optimizer,
            device=device, is_classification=is_classification
        )
        va_loss, va_metric = run_epoch(
            model=model, loader=val_loader, criterion=criterion, optimizer=None,
            device=device, is_classification=is_classification
        )

        curves["train_loss"].append(tr_loss)
        curves["val_loss"].append(va_loss)
        curves["train_metric"].append(tr_metric)
        curves["val_metric"].append(va_metric)

        # transform to "maximize" always
        if is_classification:
            if objective_mode == "loss":
                score = -va_loss
            else:
                score = va_metric  # accuracy
        else:
            score = -va_metric     # -MAE

        scheduler.step(score)

        if score > best:
            best = score
            if save_path is not None:
                torch.save(model.state_dict(), save_path)

        # optuna pruning support
        if trial is not None:
            trial.report(best, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return best, curves