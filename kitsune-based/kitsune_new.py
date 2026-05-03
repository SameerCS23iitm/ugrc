import argparse
import os
import pickle
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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
    parser.add_argument(
        "--validation-objective",
        choices=["f1", "f1_min_acc", "quantile"],
        default="f1_min_acc",
        help="How to choose the persisted validation threshold: maximize F1, maximize F1 with an accuracy floor, or use benign-score quantile.",
    )
    parser.add_argument(
        "--min-validation-accuracy",
        type=float,
        default=0.70,
        help="Minimum validation accuracy required when selecting the threshold with the F1+accuracy objective.",
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


def find_best_threshold_for_f1(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, dict]:
    """Return the threshold that maximizes F1 on labeled validation data."""
    if len(scores) == 0:
        raise ValueError("Cannot tune threshold on an empty score array.")

    candidate_thresholds = np.unique(scores)
    if candidate_thresholds.size == 1:
        candidate_thresholds = np.array(
            [candidate_thresholds[0] - 1e-12, candidate_thresholds[0], candidate_thresholds[0] + 1e-12],
            dtype=np.float64,
        )
    else:
        eps = 1e-12
        candidate_thresholds = np.concatenate(
            ([candidate_thresholds[0] - eps], candidate_thresholds, [candidate_thresholds[-1] + eps])
        )

    best_threshold = float(candidate_thresholds[0])
    best_metrics = compute_basic_metrics(y_true, scores >= best_threshold)

    for threshold in candidate_thresholds[1:]:
        metrics = compute_basic_metrics(y_true, scores >= threshold)
        if metrics["f1"] > best_metrics["f1"]:
            best_threshold = float(threshold)
            best_metrics = metrics
        elif metrics["f1"] == best_metrics["f1"]:
            if metrics["recall"] > best_metrics["recall"]:
                best_threshold = float(threshold)
                best_metrics = metrics
            elif metrics["recall"] == best_metrics["recall"] and metrics["precision"] > best_metrics["precision"]:
                best_threshold = float(threshold)
                best_metrics = metrics

    return best_threshold, best_metrics


def find_best_threshold_with_accuracy_floor(
    y_true: np.ndarray,
    scores: np.ndarray,
    min_accuracy: float,
) -> Tuple[float, dict, bool]:
    """Return the F1-best threshold that satisfies an accuracy floor when possible."""
    threshold, metrics = find_best_threshold_for_f1(y_true, scores)
    best_threshold = threshold
    best_metrics = metrics

    candidate_thresholds = np.unique(scores)
    if candidate_thresholds.size == 1:
        candidate_thresholds = np.array(
            [candidate_thresholds[0] - 1e-12, candidate_thresholds[0], candidate_thresholds[0] + 1e-12],
            dtype=np.float64,
        )
    else:
        eps = 1e-12
        candidate_thresholds = np.concatenate(
            ([candidate_thresholds[0] - eps], candidate_thresholds, [candidate_thresholds[-1] + eps])
        )

    found_feasible = False
    for candidate in candidate_thresholds:
        candidate_metrics = compute_basic_metrics(y_true, scores >= candidate)
        if candidate_metrics["accuracy"] < min_accuracy:
            continue
        if not found_feasible:
            best_threshold = float(candidate)
            best_metrics = candidate_metrics
            found_feasible = True
            continue
        if candidate_metrics["f1"] > best_metrics["f1"]:
            best_threshold = float(candidate)
            best_metrics = candidate_metrics
        elif candidate_metrics["f1"] == best_metrics["f1"]:
            if candidate_metrics["accuracy"] > best_metrics["accuracy"]:
                best_threshold = float(candidate)
                best_metrics = candidate_metrics
            elif candidate_metrics["accuracy"] == best_metrics["accuracy"] and candidate_metrics["recall"] > best_metrics["recall"]:
                best_threshold = float(candidate)
                best_metrics = candidate_metrics

    return best_threshold, best_metrics, found_feasible


def _print_metrics(m: dict) -> None:
    print(f"  accuracy : {m['accuracy']:.4f}")
    print(f"  precision: {m['precision']:.4f}")
    print(f"  recall   : {m['recall']:.4f}")
    print(f"  f1       : {m['f1']:.4f}")
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], dtype=np.int64)
    print("  confusion matrix (rows=true [0,1], cols=pred [0,1]):")
    print(cm)


def _save_confusion_matrix_plot(
    m: dict,
    tag: str,
    scenario: str,
    out_dir: str = None,
) -> str:
    """Generate and save a colored confusion matrix heatmap as PNG.

    PNGs default to the kitsune-based/obs directory.
    """
    if out_dir is None:
        out_dir = resolve_repo_path("kitsune-based/obs")
    os.makedirs(out_dir, exist_ok=True)

    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], dtype=np.int64)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    # Create heatmap with colors
    im = ax.imshow(cm, cmap="Blues", aspect="auto")

    # Set ticks and labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: Benign", "Pred: Attack"])
    ax.set_yticklabels(["True: Benign", "True: Attack"])

    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title(f"Confusion Matrix - {tag} ({scenario})", fontsize=14, fontweight="bold")

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=14,
                fontweight="bold",
            )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Count", fontsize=11, fontweight="bold")

    # Add metrics as text
    metrics_text = f"Accuracy: {m['accuracy']:.4f}\nPrecision: {m['precision']:.4f}\nRecall: {m['recall']:.4f}\nF1: {m['f1']:.4f}"
    fig.text(
        0.99,
        0.01,
        metrics_text,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    out_path = os.path.join(out_dir, f"new/cm_{tag.lower()}_{scenario}.png")
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()

    return out_path



def _score_files(
    detector,
    paths: List[str],
    feature_cols: List[str],
    threshold: float,
    tag: str,
    scenario: str,
    save_outputs: bool = False,
    out_dir: str = "",
    png_out_dir: str = None,
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
            # Save confusion matrix plot into png_out_dir (defaults to kitsune-based/obs)
            cm_plot_path = _save_confusion_matrix_plot(metrics, tag, scenario, png_out_dir)
            print(f"Saved: {cm_plot_path}")

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
    # Load pre-trained detector and retune threshold on labeled validation data.
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

    print("\nCombined validation metrics with pre-trained threshold")
    y_pred = (all_scores >= threshold).astype(np.int64)
    pretrained_metrics = compute_basic_metrics(all_labels, y_pred)
    _print_metrics(pretrained_metrics)
    # CSV outputs for kitnet scores go to kitsune-based/observations
    csv_out_dir = resolve_repo_path("kitsune-based/observations")
    os.makedirs(csv_out_dir, exist_ok=True)
    # PNG outputs go to kitsune-based/obs
    png_out_dir = resolve_repo_path("kitsune-based/obs")
    os.makedirs(png_out_dir, exist_ok=True)
    cm_plot_path = _save_confusion_matrix_plot(pretrained_metrics, "Validation-Pretrained", args.scenario, png_out_dir)
    print(f"Saved: {cm_plot_path}")

    # Fine-tune the threshold based on labeled validation data.
    if args.validation_objective == "f1":
        tuned, tuned_metrics = find_best_threshold_for_f1(all_labels, all_scores)
        print(f"\nBest validation threshold for F1: {tuned:.8f}")
        _print_metrics(tuned_metrics)
        cm_plot_path = _save_confusion_matrix_plot(tuned_metrics, "Validation-Tuned", args.scenario, png_out_dir)
        print(f"Saved: {cm_plot_path}")
        save_model(detector, feature_cols, tuned, args.model_dir)
    elif args.validation_objective == "f1_min_acc":
        tuned, tuned_metrics, feasible = find_best_threshold_with_accuracy_floor(
            all_labels,
            all_scores,
            args.min_validation_accuracy,
        )
        if feasible:
            print(
                f"\nBest validation threshold for F1 with accuracy >= {args.min_validation_accuracy:.2f}: {tuned:.8f}"
            )
        else:
            print(
                f"\nNo threshold reached accuracy >= {args.min_validation_accuracy:.2f}; falling back to pure F1 optimum."
            )
        _print_metrics(tuned_metrics)
        cm_plot_path = _save_confusion_matrix_plot(tuned_metrics, "Validation-Tuned", args.scenario, png_out_dir)
        print(f"Saved: {cm_plot_path}")
        save_model(detector, feature_cols, tuned, args.model_dir)
    else:
        benign_val_scores = all_scores[all_labels == 0]
        if len(benign_val_scores) > 0:
            tuned = float(np.quantile(benign_val_scores, args.threshold_quantile))
            print(
                f"\nRe-tuned threshold on val-benign "
                f"(q={args.threshold_quantile:.4f}): {tuned:.8f}"
            )
            print("Combined validation metrics with re-tuned threshold")
            y_pred_tuned = (all_scores >= tuned).astype(np.int64)
            tuned_metrics_quantile = compute_basic_metrics(all_labels, y_pred_tuned)
            _print_metrics(tuned_metrics_quantile)
            cm_plot_path = _save_confusion_matrix_plot(tuned_metrics_quantile, "Validation-Tuned", args.scenario, png_out_dir)
            print(f"Saved: {cm_plot_path}")
            save_model(detector, feature_cols, tuned, args.model_dir)
        else:
            print(
                "No benign samples found in validation set — "
                "keeping original threshold."
            )


def run_test(args: argparse.Namespace) -> None:
    # Test mode always uses the pre-trained model as-is; no re-training or re-tuning.
    detector, feature_cols, threshold = load_model(args.model_dir)
    print(f"[scenario={args.scenario}] Using threshold: {threshold:.8f}")

    # CSV outputs for kitnet scores go to kitsune-based/observations
    csv_out_dir = resolve_repo_path("kitsune-based/observations")
    os.makedirs(csv_out_dir, exist_ok=True)
    # PNG outputs go to kitsune-based/obs
    png_out_dir = resolve_repo_path("kitsune-based/obs")
    os.makedirs(png_out_dir, exist_ok=True)

    print("Scoring test files...")
    all_scores, all_labels = _score_files(
        detector,
        args.test_files,
        feature_cols,
        threshold,
        tag="Test",
        scenario=args.scenario,
        save_outputs=True,
        out_dir=csv_out_dir,
        png_out_dir=png_out_dir,
    )

    print("\nCombined test metrics")
    y_pred = (all_scores >= threshold).astype(np.int64)
    test_metrics = compute_basic_metrics(all_labels, y_pred)
    _print_metrics(test_metrics)
    cm_plot_path = _save_confusion_matrix_plot(test_metrics, "Test", args.scenario, png_out_dir)
    print(f"Saved: {cm_plot_path}")

    combined_path = os.path.join(csv_out_dir, f"kitnet_scores_combined_{args.scenario}.csv")
    pd.DataFrame({"label": all_labels, "score": all_scores, "pred": y_pred}).to_csv(combined_path, index=False)
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

    # For validate/test we require a previously-trained model to exist; fail fast
    if args.mode in ("validate", "test"):
        det_path = os.path.join(args.model_dir, "detector.pkl")
        if not os.path.exists(det_path):
            raise SystemExit(
                f"No trained model found at '{args.model_dir}'. Run with --mode train first or pass --model-dir pointing to a trained model."
            )
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