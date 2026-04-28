import pandas as pd
import numpy as np
import glob
import os

def load_all_logs(log_root):
    all_files = glob.glob(os.path.join(log_root, "**", "*.csv"), recursive=True)
    dfs = []
    for f in all_files:
        if "original" in os.path.basename(f).lower():  # skip corrupted file
            print(f"Skipping: {f}")
            continue
        dfs.append(pd.read_csv(f))
    return pd.concat(dfs, ignore_index=True)

DATA_DIR = "../data/"
TRAIN_DIR = "../train/"

FILES = {
    "1s": {
        "benign"   : TRAIN_DIR + "benign_flows_kit.csv",
        "cscada"   : TRAIN_DIR + "cscada_flows_kit.csv",
        "external" : TRAIN_DIR + "external_flows_kit.csv",
    }
}

def label_benign(df):
    # 1) For benign, label everything as benign by default.
    return np.zeros(len(df), dtype=int)

def _cols_for_attacker(df, attacker_ip):
    # handle columns that encode IPs with dots or underscores and contain packet count info
    dot_ip = attacker_ip
    unders_ip = attacker_ip.replace(".", "_")
    cols = [
        col for col in df.columns
        if ("packet_count" in col.lower() or "packets" in col.lower() or "tx" in col.lower())
        and (col.startswith(unders_ip) or col.startswith(dot_ip) or unders_ip in col or dot_ip in col)
    ]
    return cols

def label_cscada(df, granularity=None):
    # 2) For cscada, label windows with non-zero transactions from attacker 185.175.0.3 as attack
    attacker_ip = "185.175.0.3"
    attacker_cols = _cols_for_attacker(df, attacker_ip)
    if not attacker_cols:
        return np.zeros(len(df), dtype=int)
    return (df[attacker_cols].sum(axis=1) > 0).astype(int).values

def label_external(df):
    # 2) For external, label windows with non-zero transactions from attacker 185.175.0.7 as attack
    attacker_ip = "185.175.0.7"
    attacker_cols = _cols_for_attacker(df, attacker_ip)
    if not attacker_cols:
        return np.zeros(len(df), dtype=int)
    return (df[attacker_cols].sum(axis=1) > 0).astype(int).values

labelers = {
    "benign"   : label_benign,
    "cscada"   : label_cscada,
    "external" : label_external,
}

for gran in ["1s"]:
    print(f"\n{'='*40}")
    print(f"Granularity: {gran}")
    for scenario, filepath in FILES[gran].items():
        df = pd.read_csv(filepath)
        
        if scenario == "cscada":
            df["label"] = label_cscada(df, gran)
        elif scenario == "external":
            df["label"] = label_external(df)
        else:
            df["label"] = label_benign(df)

        total    = len(df)
        n_attack = int(df["label"].sum())
        print(f"\n  [{scenario}]")
        print(f"  Total  : {total}")
        print(f"  Attack : {n_attack} ({100*n_attack/total:.1f}%)")
        print(f"  Benign : {total-n_attack} ({100*(total-n_attack)/total:.1f}%)")

        out_path = TRAIN_DIR + f"labeled_{gran}_{scenario}.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved  → {out_path}")
