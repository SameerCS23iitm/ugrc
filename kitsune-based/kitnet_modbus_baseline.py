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
        "--validation-files",
        nargs="+",
        default=["train/labeled_1s_cscada_val.csv", "train/labeled_1s_external_val.csv"],
        help="One or more labeled CSVs used to tune the anomaly threshold.",
    )
    parser.add_argument(
        "--test-files",
        nargs="+",
        default=["train/labeled_1s_cscada_test.csv", "train/labeled_1s_external_test.csv"],
        help="One or more labeled CSVs used for final held-out evaluation.",
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
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--hidden-ratio",  type=float, default=0.75)
    parser.add_argument(
        "--mode",
        choices=["full", "train-only", "validate-only", "test-only"],
        default="full",
        help="Phase to run: 'full' trains then evals; 'train-only' trains and saves detector; "
             "'validate-only' loads detector and tunes threshold on validation; "
             "'test-only' loads detector and threshold, evaluates on test only.",
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


def print_confusion_matrix(metrics: dict, indent: str = "  ") -> None:
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]], dtype=np.int64)
    print(f"{indent}confusion matrix (rows=true [0,1], cols=pred [0,1]):")
    print(f"{indent}        pred=0  pred=1")
    print(f"{indent}true=0  {cm[0,0]:8d} {cm[0,1]:8d}")
    print(f"{indent}true=1  {cm[1,0]:8d} {cm[1,1]:8d}")


def save_detector(detector: kit.KitNET, granularity: str) -> str:
    out_dir = resolve_repo_path("observations")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"kitnet_detector_{granularity}.pkl")
    with open(path, "wb") as f:
        pickle.dump(detector, f)
    print(f"Saved trained detector to: {path}")
    return path


def load_detector(granularity: str) -> kit.KitNET:
    out_dir = resolve_repo_path("observations")
    path = os.path.join(out_dir, f"kitnet_detector_{granularity}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Detector not found at {path}. Run with --mode train-only first.")
    with open(path, "rb") as f:
        detector = pickle.load(f)
    print(f"Loaded trained detector from: {path}")
    return detector


def save_threshold(threshold: float, granularity: str) -> str:
    out_dir = resolve_repo_path("observations")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"kitnet_threshold_{granularity}.pkl")
    with open(path, "wb") as f:
        pickle.dump(threshold, f)
    print(f"Saved threshold to: {path}")
    return path


def load_threshold(granularity: str) -> float:
    out_dir = resolve_repo_path("observations")
    path = os.path.join(out_dir, f"kitnet_threshold_{granularity}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Threshold not found at {path}. Run with --mode validate-only first.")
    with open(path, "rb") as f:
        threshold = pickle.load(f)
    print(f"Loaded threshold from: {path} (value: {threshold:.8f})")
    return threshold


def score_dataset(
    detector: kit.KitNET,
    df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    part_df = df.sort_values("time_window").reset_index(drop=True)
    x_part = prepare_matrix(part_df, feature_cols)
    y_part = part_df["label"].to_numpy(dtype=np.int64)
    scores = np.array([detector.execute(x_part[i]) for i in range(len(x_part))], dtype=np.float64)
    return part_df, y_part, scores


def score_and_report_dataset(
    detector: kit.KitNET,
    df: pd.DataFrame,
    feature_cols: List[str],
    threshold: float,
    label: str,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    scored_df, labels, scores = score_dataset(detector, df, feature_cols)
    preds = (scores >= threshold).astype(np.int64)
    metrics = compute_basic_metrics(labels, preds)

    print(f"\n{label} metrics")
    print(f"  accuracy : {metrics['accuracy']:.4f}")
    print(f"  precision: {metrics['precision']:.4f}")
    print(f"  recall   : {metrics['recall']:.4f}")
    print(f"  f1       : {metrics['f1']:.4f}")
    print_confusion_matrix(metrics, indent="  ")

    return scored_df, labels, scores, metrics


def tune_threshold_from_validation(
    val_scores: np.ndarray,
    val_labels: np.ndarray,
) -> Tuple[float, dict]:
    if len(val_scores) == 0:
        raise ValueError("Validation set is empty; cannot tune threshold.")

    candidate_thresholds = np.unique(np.quantile(val_scores, np.linspace(0.01, 0.99, 99)))
    best_threshold = float(candidate_thresholds[0])
    best_metrics = compute_basic_metrics(val_labels, (val_scores >= best_threshold).astype(np.int64))

    for threshold in candidate_thresholds[1:]:
        preds = (val_scores >= threshold).astype(np.int64)
        metrics = compute_basic_metrics(val_labels, preds)
        if metrics["f1"] > best_metrics["f1"]:
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


def train_detector(args: argparse.Namespace, benign_df: pd.DataFrame, eval_parts: List[pd.DataFrame], 
                   feature_cols: List[str], x_benign: np.ndarray, x_eval: np.ndarray) -> Tuple[kit.KitNET, float]:
    """Train detector on benign data and compute initial threshold."""
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
        learning_rate=args.learning_rate,
        hidden_ratio=args.hidden_ratio,
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
    return detector, threshold


def main() -> None:
    args = parse_args()

    default_train, default_eval = choose_default_files(args)
    train_path = args.train_benign if args.train_benign else default_train
    eval_paths = args.eval_files if args.eval_files else default_eval

    # --- Train-only mode: train and save detector, then exit ---
    if args.mode == "train-only":
        print("Loading benign dataset for training...")
        benign_df = read_csv(train_path).sort_values("time_window").reset_index(drop=True)
        eval_parts = [read_csv(p) for p in eval_paths]
        feature_cols = align_feature_columns(benign_df, eval_parts)
        x_benign = prepare_matrix(benign_df, feature_cols)
        x_eval = prepare_matrix(pd.concat(eval_parts, ignore_index=True), feature_cols)
        
        detector, _ = train_detector(args, benign_df, eval_parts, feature_cols, x_benign, x_eval)
        save_detector(detector, args.granularity)
        print("Training complete. Detector saved.")
        return

    # --- Validate-only mode: load detector, tune threshold on validation, save threshold ---
    if args.mode == "validate-only":
        if not args.validation_files:
            raise ValueError("--validate-only requires --validation-files")
        
        print("Loading validation dataset...")
        val_dfs = [read_csv(p) for p in args.validation_files]
        
        # Need feature columns to load validation data; load any eval file to get them
        print(f"Loading sample eval file for feature alignment...")
        sample_eval = read_csv(eval_paths[0])
        benign_sample = read_csv(train_path)
        feature_cols = align_feature_columns(benign_sample, [sample_eval])
        
        detector = load_detector(args.granularity)
        out_dir = resolve_repo_path("observations")
        os.makedirs(out_dir, exist_ok=True)

        thresholds = []
        for val_df, path in zip(val_dfs, args.validation_files):
            val_df, y_val, val_scores = score_dataset(detector, val_df, feature_cols)
            threshold, val_metrics = tune_threshold_from_validation(val_scores, y_val)
            thresholds.append(threshold)

            print(f"Validation file: {path}")
            print(f"  threshold: {threshold:.8f}")
            print(f"  accuracy : {val_metrics['accuracy']:.4f}")
            print(f"  precision: {val_metrics['precision']:.4f}")
            print(f"  recall   : {val_metrics['recall']:.4f}")
            print(f"  f1       : {val_metrics['f1']:.4f}")
            print_confusion_matrix(val_metrics, indent="  ")

            val_out = val_df[["time_window", "label"]].copy()
            val_out["score"] = val_scores
            val_out["pred"] = (val_scores >= threshold).astype(np.int64)
            stem = os.path.splitext(os.path.basename(path))[0]
            val_path = os.path.join(out_dir, f"kitnet_scores_{stem}_{args.granularity}.csv")
            val_out.to_csv(val_path, index=False)
            print(f"  Saved validation scores to: {val_path}")

        if thresholds:
            save_threshold(float(np.median(thresholds)), args.granularity)
        return

    # --- Test-only mode: load detector and threshold, evaluate on test ---
    if args.mode == "test-only":
        if not args.test_files:
            raise ValueError("--test-only requires --test-files")
        
        print("Loading test dataset...")
        test_dfs = [read_csv(p) for p in args.test_files]
        
        # Need feature columns; load sample eval file to get them
        print(f"Loading sample eval file for feature alignment...")
        sample_eval = read_csv(eval_paths[0])
        benign_sample = read_csv(train_path)
        feature_cols = align_feature_columns(benign_sample, [sample_eval])
        
        detector = load_detector(args.granularity)
        threshold = load_threshold(args.granularity)
        
        out_dir = resolve_repo_path("observations")
        os.makedirs(out_dir, exist_ok=True)

        for test_df, path in zip(test_dfs, args.test_files):
            test_df, y_test, test_scores, test_metrics = score_and_report_dataset(
                detector,
                test_df,
                feature_cols,
                threshold,
                f"Test file: {path}",
            )

            test_out = test_df[["time_window", "label"]].copy()
            test_out["score"] = test_scores
            test_out["pred"] = (test_scores >= threshold).astype(np.int64)
            stem = os.path.splitext(os.path.basename(path))[0]
            test_path = os.path.join(out_dir, f"kitnet_scores_{stem}_{args.granularity}.csv")
            test_out.to_csv(test_path, index=False)
            print(f"  Saved test scores to: {test_path}")
        return

    # --- Full mode (default): train, then validate/test in one go ---
    print("Loading datasets...")
    benign_df = read_csv(train_path).sort_values("time_window").reset_index(drop=True)
    eval_parts = [read_csv(p) for p in eval_paths]
    eval_df = pd.concat(eval_parts, ignore_index=True)
    eval_df = eval_df.sort_values("time_window").reset_index(drop=True)

    feature_cols = align_feature_columns(benign_df, eval_parts)
    x_benign = prepare_matrix(benign_df, feature_cols)
    x_eval = prepare_matrix(eval_df, feature_cols)

    detector, threshold = train_detector(args, benign_df, eval_parts, feature_cols, x_benign, x_eval)

    if args.validation_files and args.test_files:
        print("\nValidation/test split detected; tuning threshold on validation set.")

        out_dir = resolve_repo_path("observations")
        os.makedirs(out_dir, exist_ok=True)

        thresholds = []
        for val_df, path in zip([read_csv(p) for p in args.validation_files], args.validation_files):
            val_df, y_val, val_scores = score_dataset(detector, val_df, feature_cols)
            threshold, val_metrics = tune_threshold_from_validation(val_scores, y_val)
            thresholds.append(threshold)

            print(f"Validation file: {path}")
            print(f"  threshold: {threshold:.8f}")
            print(f"  accuracy : {val_metrics['accuracy']:.4f}")
            print(f"  precision: {val_metrics['precision']:.4f}")
            print(f"  recall   : {val_metrics['recall']:.4f}")
            print(f"  f1       : {val_metrics['f1']:.4f}")
            print_confusion_matrix(val_metrics, indent="  ")

            val_out = val_df[["time_window", "label"]].copy()
            val_out["score"] = val_scores
            val_out["pred"] = (val_scores >= threshold).astype(np.int64)
            stem = os.path.splitext(os.path.basename(path))[0]
            val_path = os.path.join(out_dir, f"kitnet_scores_{stem}_{args.granularity}.csv")
            val_out.to_csv(val_path, index=False)
            print(f"  Saved validation scores to: {val_path}")

        if thresholds:
            threshold = float(np.median(thresholds))
            save_threshold(threshold, args.granularity)

        for test_df, path in zip([read_csv(p) for p in args.test_files], args.test_files):
            test_df, y_test, test_scores, test_metrics = score_and_report_dataset(
                detector,
                test_df,
                feature_cols,
                threshold,
                f"Test file: {path}",
            )

            test_out = test_df[["time_window", "label"]].copy()
            test_out["score"] = test_scores
            test_out["pred"] = (test_scores >= threshold).astype(np.int64)
            stem = os.path.splitext(os.path.basename(path))[0]
            test_path = os.path.join(out_dir, f"kitnet_scores_{stem}_{args.granularity}.csv")
            test_out.to_csv(test_path, index=False)
            print(f"  Saved test scores to: {test_path}")
        return

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
        print_confusion_matrix(metrics, indent="  ")

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
    print_confusion_matrix(combined_metrics, indent="  ")

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