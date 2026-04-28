import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KITNET_DIR = os.path.join(REPO_ROOT, "KitNET-py")
if KITNET_DIR not in sys.path:
    sys.path.append(KITNET_DIR)

import KitNET as kit  # noqa: E402


def resolve_repo_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate KitNET on labeled Modbus window datasets."
    )
    parser.add_argument(
        "--granularity",
        choices=["1s", "100ms"],
        default="1s",
        help="Window granularity to use.",
    )
    parser.add_argument(
        "--train-benign",
        default="train/labeled_1s_benign.csv",
        help="Path to benign-only CSV for unsupervised training.",
    )
    parser.add_argument(
        "--eval-files",
        nargs="+",
        default=["train/labeled_1s_cscada.csv", "train/labeled_1s_external.csv"],
        help="One or more labeled CSVs to evaluate on.",
    )
    parser.add_argument(
        "--max-ae",
        type=int,
        default=16,
        help="Maximum size of any autoencoder in KitNET ensemble.",
    )
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
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.995,
        help="Quantile from benign execution-phase scores used as anomaly threshold.",
    )
    return parser.parse_args()


def choose_default_files(args: argparse.Namespace) -> Tuple[str, List[str]]:
    if args.granularity == "100ms":
        return (
            "train/labeled_100ms_benign.csv",
            ["train/labeled_100ms_cscada.csv", "train/labeled_100ms_external.csv"],
        )
    return (
        "train/labeled_1s_benign.csv",
        ["train/labeled_1s_cscada.csv", "train/labeled_1s_external.csv"],
    )


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


def main() -> None:
    args = parse_args()

    default_train, default_eval = choose_default_files(args)
    train_path = args.train_benign if args.train_benign else default_train
    eval_paths = args.eval_files if args.eval_files else default_eval

    print("Loading datasets...")
    benign_df = read_csv(train_path).sort_values("time_window").reset_index(drop=True)
    eval_parts = [read_csv(p) for p in eval_paths]
    eval_df = pd.concat(eval_parts, ignore_index=True)
    eval_df = eval_df.sort_values("time_window").reset_index(drop=True)

    feature_cols = align_feature_columns(benign_df, eval_parts)
    x_benign = prepare_matrix(benign_df, feature_cols)
    x_eval = prepare_matrix(eval_df, feature_cols)

    if args.fm_grace is None or args.ad_grace is None:
        fm_default, ad_default = auto_grace_periods(len(x_benign))
        fm_grace = args.fm_grace if args.fm_grace is not None else fm_default
        ad_grace = args.ad_grace if args.ad_grace is not None else ad_default
    else:
        fm_grace = args.fm_grace
        ad_grace = args.ad_grace

    print("KitNET setup")
    print(f"  features: {len(feature_cols)}")
    print(f"  max_ae: {args.max_ae}")
    print(f"  FM_grace: {fm_grace}")
    print(f"  AD_grace: {ad_grace}")

    detector = kit.KitNET(
        n=len(feature_cols),
        max_autoencoder_size=args.max_ae,
        FM_grace_period=fm_grace,
        AD_grace_period=ad_grace,
    )

    print("Training on benign stream...")
    benign_scores = np.zeros(len(x_benign), dtype=np.float64)
    for i in range(len(x_benign)):
        benign_scores[i] = detector.process(x_benign[i])

    grace_end = fm_grace + ad_grace + 1
    exec_scores = benign_scores[grace_end:] if grace_end < len(benign_scores) else np.array([])
    if len(exec_scores) == 0:
        print("Warning: no benign execution-phase scores available after grace period.")
        print("Using first 100 eval rows for rough threshold warm-up.")
        warmup = min(100, len(x_eval))
        _ = [detector.execute(x_eval[i]) for i in range(warmup)]
        exec_scores = np.array([detector.execute(x_eval[i]) for i in range(warmup)], dtype=np.float64)

    threshold = float(np.quantile(exec_scores, args.threshold_quantile))
    print(f"Threshold (q={args.threshold_quantile:.4f}): {threshold:.8f}")

    print("Scoring evaluation streams...")
    eval_outputs = []
    for path, part_df in zip(eval_paths, eval_parts):
        part_df = part_df.sort_values("time_window").reset_index(drop=True)
        x_part = prepare_matrix(part_df, feature_cols)
        y_part = part_df["label"].to_numpy(dtype=np.int64)
        part_scores = np.array(
            [detector.execute(x_part[i]) for i in range(len(x_part))],
            dtype=np.float64,
        )
        part_pred = (part_scores >= threshold).astype(np.int64)

        metrics = compute_basic_metrics(y_part, part_pred)
        print(f"\nMetrics on {path}")
        print(f"  accuracy : {metrics['accuracy']:.4f}")
        print(f"  precision: {metrics['precision']:.4f}")
        print(f"  recall   : {metrics['recall']:.4f}")
        print(f"  f1       : {metrics['f1']:.4f}")
        confusion_matrix = np.array(
            [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
            dtype=np.int64,
        )
        print("  confusion matrix (rows=true [0,1], cols=pred [0,1]):")
        print(confusion_matrix)

        per_file_out = part_df[["time_window", "label"]].copy()
        per_file_out["score"] = part_scores
        per_file_out["pred"] = part_pred
        eval_outputs.append((path, per_file_out))

    y_eval = eval_df["label"].to_numpy(dtype=np.int64)
    eval_scores = np.array([detector.execute(x_eval[i]) for i in range(len(x_eval))], dtype=np.float64)
    y_pred = (eval_scores >= threshold).astype(np.int64)
    combined_metrics = compute_basic_metrics(y_eval, y_pred)
    print("\nCombined metrics across all eval files")
    print(f"  accuracy : {combined_metrics['accuracy']:.4f}")
    print(f"  precision: {combined_metrics['precision']:.4f}")
    print(f"  recall   : {combined_metrics['recall']:.4f}")
    print(f"  f1       : {combined_metrics['f1']:.4f}")

    out = eval_df[["time_window", "label"]].copy()
    out["score"] = eval_scores
    out["pred"] = y_pred
    out_dir = resolve_repo_path("observations")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"kitnet_scores_{args.granularity}.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved combined detailed scores to: {out_path}")

    for path, per_file_out in eval_outputs:
        stem = os.path.splitext(os.path.basename(path))[0]
        per_file_path = os.path.join(out_dir, f"kitnet_scores_{stem}_{args.granularity}.csv")
        per_file_out.to_csv(per_file_path, index=False)
        print(f"Saved per-file detailed scores to: {per_file_path}")


if __name__ == "__main__":
    main()
    import os
    os.system('notify-send "Python Script" "Execution complete!"')