import argparse
import os
import pickle
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KITNET_DIR = os.path.join(REPO_ROOT, "KitNET-py")
if KITNET_DIR not in sys.path:
    sys.path.append(KITNET_DIR)

import KitNET as kit  # noqa: E402


# ---------------------------------------------------------------------------
# Hardcoded file paths per scenario
# ---------------------------------------------------------------------------

TRAIN_BENIGN = "train/labeled_1s_benign.csv"

VAL_FILES = {
    "cscada":   ["train/labeled_1s_cscada_val.csv"],
    "external": ["train/labeled_1s_external_val.csv"],
}

TEST_FILES = {
    "cscada":   ["train/labeled_1s_cscada_test.csv"],
    "external": ["train/labeled_1s_external_test.csv"],
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate KitNET on labeled Modbus window datasets."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "validate", "test"],
        required=True,
        help="Pipeline phase to run: train, validate, or test.",
    )
    parser.add_argument(
        "--scenario",
        choices=["cscada", "external"],
        required=True,
        help="Which attack scenario to run (cscada or external).",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Directory to save/load model state. Defaults to observations/kitnet_model_<scenario>.",
    )
    # KitNET architecture hyperparameters
    parser.add_argument(
        "--max-ae",
        type=int,
        default=16,
        help="Maximum number of inputs to any single autoencoder in the KitNET ensemble.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate for KitNET autoencoders.",
    )
    parser.add_argument(
        "--hidden-ratio",
        type=float,
        default=0.75,
        help="Hidden layer size ratio for each autoencoder (0 < r < 1).",
    )
    # Grace period controls
    parser.add_argument(
        "--fm-grace",
        type=int,
        default=None,
        help="Feature mapping grace period. If omitted, auto-computed from training size.",
    )
    parser.add_argument(
        "--ad-grace",
        type=int,
        default=None,
        help="Anomaly detector grace period. If omitted, auto-computed from training size.",
    )
    # Threshold
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.995,
        help="Quantile from benign execution-phase scores used as anomaly threshold.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def resolve_repo_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_csv(path: str) -> pd.DataFrame:
    resolved_path = resolve_repo_path(path)
    df = pd.read_csv(resolved_path)
    if "label" not in df.columns:
        raise ValueError(f"Missing 'label' column in {resolved_path}")
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    excluded = {"label", "time_window", "ip.src", "ip.dst"}
    candidate = [c for c in df.columns if c not in excluded]
    numeric = [c for c in candidate if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        raise ValueError("No numeric feature columns found after exclusions.")
    return numeric


def align_feature_columns(train_df: pd.DataFrame, eval_dfs: List[pd.DataFrame]) -> List[str]:
    train_cols = set(get_feature_columns(train_df))
    eval_cols = [set(get_feature_columns(df)) for df in eval_dfs]
    common = train_cols
    for cols in eval_cols:
        common = common.intersection(cols)
    common_cols = sorted(common)
    if not common_cols:
        raise ValueError("No common numeric feature columns between train/eval files.")
    return common_cols


def prepare_matrix(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    x = df[feature_cols].to_numpy(dtype=np.float64)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(
    detector,
    feature_cols: List[str],
    threshold: float,
    model_dir: str,
) -> None:
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "detector.pkl"), "wb") as f:
        pickle.dump(detector, f)
    with open(os.path.join(model_dir, "meta.pkl"), "wb") as f:
        pickle.dump({"feature_cols": feature_cols, "threshold": threshold}, f)
    print(f"Model saved to: {model_dir}")


def load_model(model_dir: str):
    det_path = os.path.join(model_dir, "detector.pkl")
    meta_path = os.path.join(model_dir, "meta.pkl")
    if not os.path.exists(det_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"No saved model found in '{model_dir}'. Run --mode train first."
        )
    with open(det_path, "rb") as f:
        detector = pickle.load(f)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    print(f"Model loaded from: {model_dir}")
    return detector, meta["feature_cols"], meta["threshold"]


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def auto_grace_periods(n_train: int) -> Tuple[int, int]:
    fm = max(200, min(5000, int(0.15 * n_train)))
    ad = max(500, min(20000, int(0.60 * n_train)))
    total = fm + ad
    cap = max(20, n_train - 10)
    if total > cap:
        scale = cap / float(total)
        fm = max(10, int(fm * scale))
        ad = max(10, int(ad * scale))
    return fm, ad


def compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    acc = (tp + tn) / max(1, len(y_true))
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _print_metrics(m: dict) -> None:
    print(f"  accuracy : {m['accuracy']:.4f}")
    print(f"  precision: {m['precision']:.4f}")
    print(f"  recall   : {m['recall']:.4f}")
    print(f"  f1       : {m['f1']:.4f}")
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], dtype=np.int64)
    print("  confusion matrix (rows=true [0,1], cols=pred [0,1]):")
    print(cm)


def _score_files(
    detector,
    paths: List[str],
    feature_cols: List[str],
    threshold: float,
    tag: str,
    scenario: str,
    save_outputs: bool = False,
    out_dir: str = "",
) -> Tuple[np.ndarray, np.ndarray]:
    """Score one or more CSV files, print per-file metrics, optionally save CSVs.

    Returns concatenated (all_scores, all_labels) across all files.
    """
    all_scores, all_labels = [], []

    for path in paths:
        part_df = read_csv(path).sort_values("time_window").reset_index(drop=True)
        x_part = prepare_matrix(part_df, feature_cols)
        y_part = part_df["label"].to_numpy(dtype=np.int64)

        part_scores = np.array(
            [detector.execute(x_part[i]) for i in range(len(x_part))],
            dtype=np.float64,
        )
        part_pred = (part_scores >= threshold).astype(np.int64)
        metrics = compute_basic_metrics(y_part, part_pred)

        print(f"\n{tag} metrics on {path}  [threshold={threshold:.8f}]")
        _print_metrics(metrics)

        all_scores.append(part_scores)
        all_labels.append(y_part)

        if save_outputs and out_dir:
            stem = os.path.splitext(os.path.basename(path))[0]
            out = part_df[["time_window", "label"]].copy()
            out["score"] = part_scores
            out["pred"] = part_pred
            out_path = os.path.join(out_dir, f"kitnet_scores_{stem}_{scenario}.csv")
            out.to_csv(out_path, index=False)
            print(f"Saved: {out_path}")

    return np.concatenate(all_scores), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Phase functions
# ---------------------------------------------------------------------------

def run_train(args: argparse.Namespace) -> None:
    print(f"[scenario={args.scenario}] Loading benign training data...")
    benign_df = read_csv(args.train_benign).sort_values("time_window").reset_index(drop=True)

    # Align feature columns against val files so all phases share the same column set
    val_parts = [read_csv(p) for p in args.val_files]
    feature_cols = align_feature_columns(benign_df, val_parts)
    x_benign = prepare_matrix(benign_df, feature_cols)

    if args.fm_grace is None or args.ad_grace is None:
        fm_default, ad_default = auto_grace_periods(len(x_benign))
        fm_grace = args.fm_grace if args.fm_grace is not None else fm_default
        ad_grace = args.ad_grace if args.ad_grace is not None else ad_default
    else:
        fm_grace, ad_grace = args.fm_grace, args.ad_grace

    print("KitNET setup")
    print(f"  features      : {len(feature_cols)}")
    print(f"  max_ae        : {args.max_ae}")
    print(f"  FM_grace      : {fm_grace}")
    print(f"  AD_grace      : {ad_grace}")
    print(f"  learning_rate : {args.learning_rate}")
    print(f"  hidden_ratio  : {args.hidden_ratio}")

    detector = kit.KitNET(
        n=len(feature_cols),
        max_autoencoder_size=args.max_ae,
        FM_grace_period=fm_grace,
        AD_grace_period=ad_grace,
        learning_rate=args.learning_rate,
        hidden_ratio=args.hidden_ratio,
    )

    print("Training on benign stream...")
    benign_scores = np.zeros(len(x_benign), dtype=np.float64)
    for i in range(len(x_benign)):
        benign_scores[i] = detector.process(x_benign[i])

    grace_end = fm_grace + ad_grace + 1
    exec_scores = (
        benign_scores[grace_end:] if grace_end < len(benign_scores) else np.array([])
    )

    if len(exec_scores) == 0:
        print(
            "Warning: no execution-phase scores after grace period. "
            "Threshold set to NaN — run --mode validate to tune it."
        )
        threshold = float("nan")
    else:
        threshold = float(np.quantile(exec_scores, args.threshold_quantile))
        print(f"Initial threshold (q={args.threshold_quantile:.4f}): {threshold:.8f}")

    save_model(detector, feature_cols, threshold, args.model_dir)


def run_validate(args: argparse.Namespace) -> None:
    detector, feature_cols, threshold = load_model(args.model_dir)

    print(f"[scenario={args.scenario}] Scoring validation files...")
    all_scores, all_labels = _score_files(
        detector,
        args.val_files,
        feature_cols,
        threshold,
        tag="Validation",
        scenario=args.scenario,
    )

    print("\nCombined validation metrics")
    y_pred = (all_scores >= threshold).astype(np.int64)
    _print_metrics(compute_basic_metrics(all_labels, y_pred))

    # Re-tune threshold on val-benign scores and persist so test picks it up
    benign_val_scores = all_scores[all_labels == 0]
    if len(benign_val_scores) > 0:
        tuned = float(np.quantile(benign_val_scores, args.threshold_quantile))
        print(
            f"\nRe-tuned threshold on val-benign "
            f"(q={args.threshold_quantile:.4f}): {tuned:.8f}"
        )
        print("Combined validation metrics with re-tuned threshold")
        y_pred_tuned = (all_scores >= tuned).astype(np.int64)
        _print_metrics(compute_basic_metrics(all_labels, y_pred_tuned))
        save_model(detector, feature_cols, tuned, args.model_dir)
    else:
        print(
            "No benign samples found in validation set — "
            "keeping original threshold."
        )


def run_test(args: argparse.Namespace) -> None:
    detector, feature_cols, threshold = load_model(args.model_dir)
    print(f"[scenario={args.scenario}] Using threshold: {threshold:.8f}")

    out_dir = resolve_repo_path("observations")
    os.makedirs(out_dir, exist_ok=True)

    print("Scoring test files...")
    all_scores, all_labels = _score_files(
        detector,
        args.test_files,
        feature_cols,
        threshold,
        tag="Test",
        scenario=args.scenario,
        save_outputs=True,
        out_dir=out_dir,
    )

    print("\nCombined test metrics")
    y_pred = (all_scores >= threshold).astype(np.int64)
    _print_metrics(compute_basic_metrics(all_labels, y_pred))

    combined_path = os.path.join(
        out_dir, f"kitnet_scores_combined_{args.scenario}.csv"
    )
    pd.DataFrame(
        {"label": all_labels, "score": all_scores, "pred": y_pred}
    ).to_csv(combined_path, index=False)
    print(f"Saved combined scores to: {combined_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Resolve hardcoded file paths from scenario
    args.train_benign = TRAIN_BENIGN
    args.val_files    = VAL_FILES[args.scenario]
    args.test_files   = TEST_FILES[args.scenario]

    # Scenario-isolated model directory
    if args.model_dir is None:
        args.model_dir = resolve_repo_path(f"observations/kitnet_model_{args.scenario}")
    else:
        args.model_dir = resolve_repo_path(args.model_dir)

    dispatch = {
        "train":    run_train,
        "validate": run_validate,
        "test":     run_test,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
    import os
    os.system('notify-send "Python Script" "Execution complete!"')