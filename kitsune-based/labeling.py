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
LOG_ROOT = DATA_DIR + "attack/compromised-scada/attack logs/"
ATTACKER_IP = "185.175.0.7"

FILES = {
    "1s": {
        "benign"   : TRAIN_DIR + "benign_flows_kit.csv",
        "cscada"   : TRAIN_DIR + "cscada_flows_kit.csv",
        "external" : TRAIN_DIR + "external_flows_kit.csv",
    }
}

TOLERANCE = {"1s": 10} 

logs = load_all_logs(LOG_ROOT)
logs["timestamp"] = pd.to_datetime(logs["Timestamp"], utc=True, errors="coerce")
logs["unix_ts"] = logs["timestamp"].astype(np.int64) // 10**6
attack_times = logs["unix_ts"].values
print(f"Loaded {len(logs)} attack log entries")
print(f"Time range: {logs['Timestamp'].min()} → {logs['Timestamp'].max()}")

df = pd.read_csv(TRAIN_DIR + "1s_cscada_flows.csv")
print("Flow time range:")
print(pd.to_datetime(df["time_window"], unit='s').min())
print(pd.to_datetime(df["time_window"], unit='s').max())

print("\nAttack log time range:")
print(logs["Timestamp"].min())
print(logs["Timestamp"].max())

def label_cscada(df, granularity):
    tol = TOLERANCE[granularity]
    ts = df["time_window"].values.astype(np.float64)
   
    sorted_atk = np.sort(attack_times)
    
    # For each ts, find closest attack time using binary search
    idx = np.searchsorted(sorted_atk, ts)
    
    # Check neighbor to the left and right
    left  = np.where(idx > 0, sorted_atk[np.clip(idx-1, 0, len(sorted_atk)-1)], np.inf)
    right = np.where(idx < len(sorted_atk), sorted_atk[np.clip(idx, 0, len(sorted_atk)-1)], np.inf)
    
    min_dist = np.minimum(np.abs(ts - left), np.abs(ts - right))
    return (min_dist <= tol).astype(int)

def label_external(df):
    # attacker_col_prefix = ATTACKER_IP.replace(".", "_")
    # attacker_cols = [
    #     col for col in df.columns
    #     if col.startswith(attacker_col_prefix) and "packet_count" in col
    # ]
    # return (df[attacker_cols].sum(axis=1) > 0).astype(int).values
    logs = load_all_logs(DATA_DIR + "attack/external/external-attacker/attacker logs/")
    logs["timestamp"] = pd.to_datetime(logs["Timestamp"], utc=True, errors="coerce")
    logs["unix_ts"] = logs["timestamp"].astype(np.int64) // 10**6
    attack_times = logs["unix_ts"].values
    print(f"Loaded {len(logs)} attack log entries")
    print(f"Time range: {logs['Timestamp'].min()} → {logs['Timestamp'].max()}")
    tol = TOLERANCE['1s']
    ts = df["time_window"].values.astype(np.float64)
   
    sorted_atk = np.sort(attack_times)
    
    # For each ts, find closest attack time using binary search
    idx = np.searchsorted(sorted_atk, ts)
    
    # Check neighbor to the left and right
    left  = np.where(idx > 0, sorted_atk[np.clip(idx-1, 0, len(sorted_atk)-1)], np.inf)
    right = np.where(idx < len(sorted_atk), sorted_atk[np.clip(idx, 0, len(sorted_atk)-1)], np.inf)
    
    min_dist = np.minimum(np.abs(ts - left), np.abs(ts - right))
    return (min_dist <= tol).astype(int)

def label_benign(df):
    return np.zeros(len(df), dtype=int)

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
        n_attack = df["label"].sum()
        print(f"\n  [{scenario}]")
        print(f"  Total  : {total}")
        print(f"  Attack : {n_attack} ({100*n_attack/total:.1f}%)")
        print(f"  Benign : {total-n_attack} ({100*(total-n_attack)/total:.1f}%)")

        out_path = TRAIN_DIR + f"labeled_{gran}_{scenario}.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved  → {out_path}")



# import pandas as pd
# import numpy as np
# import glob
# import os

# def load_all_logs(log_root):
#     all_files = glob.glob(os.path.join(log_root, "**", "*.csv"), recursive=True)
#     dfs = []
#     for f in all_files:
#         if "original" in os.path.basename(f).lower():
#             print(f"Skipping: {f}")
#             continue
#         dfs.append(pd.read_csv(f))
#     return pd.concat(dfs, ignore_index=True)

# DATA_DIR = "../data/"
# TRAIN_DIR = "../train/"
# LOG_ROOT = DATA_DIR + "attack/compromised-scada/attack logs/"
# ATTACKER_IP = "185.175.0.7"

# FILES = {
#     "1s": {
#         "benign"   : TRAIN_DIR + "benign_flows_kit.csv",
#         "cscada"   : TRAIN_DIR + "cscada_flows_kit.csv",
#         "external" : TRAIN_DIR + "external_flows_kit.csv",
#     }
# }

# TOLERANCE_MIN = {"1s": 500}  # seconds
# TOLERANCE_MAX = {"1s": 800}  # seconds

# logs = load_all_logs(LOG_ROOT)
# logs["timestamp"] = pd.to_datetime(logs["Timestamp"], utc=True, errors="coerce")
# logs["unix_ts"] = logs["timestamp"].astype(np.int64) // 10**6  # microseconds → seconds
# attack_times = logs["unix_ts"].values

# print(f"Loaded {len(logs)} attack log entries")
# print(f"Time range: {logs['Timestamp'].min()} → {logs['Timestamp'].max()}")


# def label_cscada(df, granularity):
#     tol_min = TOLERANCE_MIN[granularity]
#     tol_max = TOLERANCE_MAX[granularity]

#     ts = df["time_window"].values.astype(np.float64)
#     labels = np.zeros(len(ts), dtype=int)
    
#     start_time = ts[0]

#     for atk_t in attack_times:
#         # Before the attack: 500-800s before
#         lo_before = int(atk_t - start_time - tol_max)
#         hi_before = int(atk_t - start_time - tol_min)

#         # After the attack: 500-800s after
#         lo_after = int(atk_t - start_time + tol_min)
#         hi_after = int(atk_t - start_time + tol_max)

#         for lo, hi in [(lo_before, hi_before), (lo_after, hi_after)]:
#             lo = max(lo, 0)
#             hi = min(hi, len(labels) - 1)
#             if lo <= hi:
#                 labels[lo:hi+1] = 1

#     return labels

# def label_external(df):
#     attacker_col_prefix = ATTACKER_IP.replace(".", "_")
#     attacker_cols = [
#         col for col in df.columns
#         if col.startswith(attacker_col_prefix) and "packet_count" in col
#     ]
#     return (df[attacker_cols].sum(axis=1) > 0).astype(int).values


# def label_benign(df):
#     return np.zeros(len(df), dtype=int)


# for gran in ["1s"]:
#     print(f"\n{'='*40}")
#     print(f"Granularity: {gran}")

#     for scenario, filepath in FILES[gran].items():
#         df = pd.read_csv(filepath)

#         if scenario == "cscada":
#             df["label"] = label_cscada(df, gran)
#         elif scenario == "external":
#             df["label"] = label_external(df)
#         else:
#             df["label"] = label_benign(df)

#         total    = len(df)
#         n_attack = df["label"].sum()
#         print(f"\n  [{scenario}]")
#         print(f"  Total  : {total}")
#         print(f"  Attack : {n_attack} ({100*n_attack/total:.1f}%)")
#         print(f"  Benign : {total-n_attack} ({100*(total-n_attack)/total:.1f}%)")

#         out_path = TRAIN_DIR + f"labeled_{gran}_{scenario}.csv"
#         df.to_csv(out_path, index=False)
#         print(f"  Saved  → {out_path}")