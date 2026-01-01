import argparse
from typing import Tuple
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import optuna
from optuna.trial import Trial
import pandas as pd
import numpy as np

from lstm import LSTMPredictor
from features import (
    all_feature_fn,
    numerical_feature_fn,
    embedding_feature_fn,
    target_feature_fn,
    garbage_feature_fn)
from sequence_dataset import build_sequence_dataset
from pca import apply_pca_to_embeddings
from random_utils.visualization import save_curves
from training import train_eval


parser = argparse.ArgumentParser(description="Train LSTM on LoL match history sequences")
# Data and target parameters
parser.add_argument("--dataset", type=str, default="m-player+team_asr-corr_mdl-openai-oai_emb3_ck-t512-o256.parquet",
                    help="Path to dataset parquet file")
parser.add_argument("--feature_fn", type=str, default="embedding",
                    choices=["embedding", "target", "numerical", "all", "garbage"],
                    help="Which feature function to use")
parser.add_argument("--target_col", type=str, default="team1_result",
                    choices=["team1_result", "team1_kills", "gamelength", "kill_total"], )
# Run quantity parameters
parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
parser.add_argument("--runs", type=int, default=1, help="How many times to run full training.")
# Sequence modeling parameters
parser.add_argument("--k", type=int, default=8, help="Number of history matches per team")
parser.add_argument("--min_history", type=int, default=4, help="Minimum history required for a sample")
# LSTM parameters
parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension of LSTM")
parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate in LSTM")
parser.add_argument("--bidirectional", action="store_true", help="Use bidirectional LSTM")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW")
parser.add_argument("--pooling", type=str, default="last", choices=["last", "mean"], help="LSTM pooling mode")
# Preprocessing parameter
parser.add_argument(
    "--pca",
    type=int,
    default=0,
    help="If > 0, apply PCA to the embedding column to this dimensionality before building the dataset.",
)

parser.add_argument("--optuna_run", action="store_true", help="Run Optuna hyperparam search before final runs.")
parser.add_argument("--optuna_trials", type=int, default=30, help="Number of Optuna trials.")
parser.add_argument("--optuna_epochs", type=int, default=10, help="Epochs per Optuna trial (cheap search).")
parser.add_argument("--optuna_timeout", type=int, default=0, help="Timeout in seconds (0 = no timeout).")
parser.add_argument("--optuna_seed", type=int, default=42, help="Seed for Optuna sampler.")
parser.add_argument("--optuna_save", type=str, default="optuna_best.json", help="Where to save best params as JSON.")
parser.add_argument("--optuna_metric", type=str, default="acc", choices=["acc", "loss"],
                    help="Objective metric for classification (acc=max, loss=min). For regression it uses MAE=min.")

class SeqDataset(Dataset):
    """
    Torch Dataset wrapper for fixed-length sequence tensors and corresponding targets.
    """
    def __init__(self, X_arr: np.ndarray, y_arr: np.ndarray):
        self.X = torch.from_numpy(X_arr)
        self.y = torch.from_numpy(y_arr)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]

# --- helper: global seeding for reproducibility ---
def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# --- helper: build model/criterion/optimizer/scheduler from params ---
def build_training_objects(
    *,
    input_dim: int,
    device: torch.device,
    is_classification: bool,
    y_train_np: np.ndarray,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    bidirectional: bool,
    pooling: str,
    lr: float,
    weight_decay: float,
):
    """
    Construct the model, loss function, optimizer, and learning-rate scheduler
    for a single training run.
    """
    model = LSTMPredictor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        pooling=pooling,
    ).to(device)

    if is_classification:
        pos = int((y_train_np == 1).sum())
        neg = int((y_train_np == 0).sum())
        pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",      # we pass transformed metric (acc or -MAE / -loss)
        factor=0.9,
        patience=3,
        min_lr=1e-5,
    )
    return model, criterion, optimizer, scheduler



if __name__ == "__main__":
    args = parser.parse_args()
    seed_everything(args.optuna_seed)

    # --- feature function selection (late-bind args.target_col safely)
    if args.feature_fn == "embedding":
        feat_fn = embedding_feature_fn
    elif args.feature_fn == "garbage":
        feat_fn = lambda row, team: garbage_feature_fn()
    elif args.feature_fn == "target":
        feat_fn = lambda row, team: target_feature_fn(row, team, args.target_col)
    elif args.feature_fn == "numerical":
        feat_fn = lambda row, team: numerical_feature_fn(row, team)
    elif args.feature_fn == "all":
        feat_fn = lambda row, team: all_feature_fn(row, team)
    else:
        raise ValueError(f"Unknown feature_fn={args.feature_fn}")

    print(
        f"Using feature_fn={args.feature_fn}, k={args.k}, "
        f"hidden_dim={args.hidden_dim}, layers={args.num_layers}, pooling={args.pooling}, "
        f"bidirectional={args.bidirectional}"
    )
    print(f"Predicting target_col={args.target_col}")

    # --- load + enrich
    df = pd.read_parquet(args.dataset)
    df = df.sort_values("date")
    print("Original raw data size: ", len(df))
    df["kill_diff"] = (df["team1_kills"] - df["team2_kills"]).abs()
    df["kill_total"] = df["team1_kills"] + df["team2_kills"]

    # --- optional PCA
    df = apply_pca_to_embeddings(df, emb_col="embedding", n_components=args.pca)

    # --- dataset build (chronological by builder)
    data = build_sequence_dataset(
        df,
        k_history=args.k,
        min_history=args.min_history,
        pad_to_k=True,
        feature_fn=feat_fn,
        target_column=args.target_col,
    )

    X = data["X"].astype(np.float32)
    y_raw = data["y"]
    print("Samples:", X.shape, "Targets:", y_raw.shape)
    print("First sample meta:", data["meta"][0])

    is_classification = (args.target_col == "team1_result")
    y = y_raw.astype(np.float32)

    # --- chronological split
    N = X.shape[0]
    split = int(0.8 * N)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    # --- data loaders (rebuilt per trial so batch_size can vary)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def make_loaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(
            SeqDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            SeqDataset(X_val, y_val),
            batch_size=batch_size * 2,
            shuffle=False,
            drop_last=False,
        )
        return train_loader, val_loader

    # -------------------------
    # Optuna search (optional)
    # -------------------------
    if args.optuna_run:
        # Note: we maximize `score` defined in train_eval (acc or -loss / -MAE)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        sampler = optuna.samplers.TPESampler(seed=args.optuna_seed)

        def objective(trial: Trial) -> float:
            # define search ranges
            hidden_dim = trial.suggest_categorical("hidden_dim", [64, 96, 128, 192, 256, 384, 512])
            num_layers = trial.suggest_int("num_layers", 1, 4)
            dropout = trial.suggest_float("dropout", 0.0, 0.35)
            bidirectional = trial.suggest_categorical("bidirectional", [False])
            pooling = trial.suggest_categorical("pooling", ["last", "mean"])

            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-3, log=True)
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 96, 128])
            train_loader, val_loader = make_loaders(batch_size)

            model, criterion, optimizer, scheduler = build_training_objects(
                input_dim=X.shape[2],
                device=device,
                is_classification=is_classification,
                y_train_np=y_train,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
                pooling=pooling,
                lr=lr,
                weight_decay=weight_decay,
            )

            # temp checkpoint per trial (optional)
            ckpt_path = None  # or f"optuna_trial_{trial.number}.pt"

            score, _ = train_eval(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                is_classification=is_classification,
                epochs=args.optuna_epochs,
                save_path=ckpt_path,
                trial=trial,
                objective_mode=args.optuna_metric,
            )
            return score

        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
        study.optimize(
            objective,
            n_trials=args.optuna_trials,
            timeout=None if args.optuna_timeout <= 0 else args.optuna_timeout,
            show_progress_bar=True,
        )

        print("Optuna best score:", study.best_value)
        print("Optuna best params:", study.best_params)

        # overwrite args.* with best params (so the final args.runs use them)
        best = study.best_params
        args.hidden_dim = int(best.get("hidden_dim", args.hidden_dim))
        args.num_layers = int(best.get("num_layers", args.num_layers))
        args.dropout = float(best.get("dropout", args.dropout))
        args.bidirectional = bool(best.get("bidirectional", args.bidirectional))
        args.pooling = str(best.get("pooling", args.pooling))
        args.lr = float(best.get("lr", args.lr))
        args.weight_decay = float(best.get("weight_decay", args.weight_decay))
        args.batch_size = int(best.get("batch_size", args.batch_size))

        # save best params to json for reproducibility
        payload = {
            "best_score": float(study.best_value),
            "best_params": study.best_params,
            "optuna_trials": args.optuna_trials,
            "optuna_epochs": args.optuna_epochs,
            "objective": args.optuna_metric,
        }
        with open(args.optuna_save, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved Optuna best params to {args.optuna_save}")

        # rebuild loaders with tuned batch size
        train_loader, val_loader = make_loaders(args.batch_size)

    else:
        train_loader, val_loader = make_loaders(args.batch_size)

    # -------------------------
    # Final runs (args.runs)
    # -------------------------
    best_metrics = []
    for run in range(args.runs):
        model, criterion, optimizer, scheduler = build_training_objects(
            input_dim=X.shape[2],
            device=device,
            is_classification=is_classification,
            y_train_np=y_train,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
            pooling=args.pooling,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        ckpt = f"lstm_best_{args.feature_fn}_{args.target_col}_run{run+1}.pt"
        best_val_metric, curves = train_eval(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            is_classification=is_classification,
            epochs=args.epochs,
            save_path=ckpt,
            trial=None,
            objective_mode=args.optuna_metric,
        )

        # save plots
        epochs_arr = np.arange(1, args.epochs + 1)
        save_curves(
            epochs=epochs_arr,
            train_losses=curves["train_loss"],
            val_losses=curves["val_loss"],
            train_metrics=curves["train_metric"],
            val_metrics=curves["val_metric"],
            is_classification=is_classification,
            feature_fn=args.feature_fn,
            target_col=args.target_col,
        )

        print(f"Run {run+1}/{args.runs} complete. Best val score: {best_val_metric:.3f}")
        best_metrics.append(best_val_metric)

    vals = np.array(best_metrics, dtype=float)
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    print(f"mean = {mean:.6f}, std = {std:.6f}")
