import pandas as pd
import numpy as np

WINDOW_SIZE = 20
STRIDE = 5

df = pd.read_csv("../data/test/attack_ft.csv")

# Explicit numeric columns (future-proof)
num_cols = [
    "iat",
    "frame.len",
    "tcp.len",
    "tcp.flags",
    "mbtcp.len",
    "modbus.func_code",
]

df = df[num_cols]
df = df.apply(pd.to_numeric, errors="coerce")
df = df.dropna()

windows = []

for i in range(0, len(df) - WINDOW_SIZE + 1, STRIDE):
    w = df.iloc[i : i + WINDOW_SIZE]

    features = {
        "iat_mean": w["iat"].mean(),
        "iat_std": w["iat"].std(),
        "iat_max": w["iat"].max(),

        "frame_len_mean": w["frame.len"].mean(),
        "frame_len_std": w["frame.len"].std(),
        "frame_len_max": w["frame.len"].max(),

        "tcp_len_mean": w["tcp.len"].mean(),
        "tcp_len_std": w["tcp.len"].std(),

        "tcp_flags_unique": w["tcp.flags"].nunique(),

        "modbus_func_unique": w["modbus.func_code"].nunique(),
        "modbus_ratio": w["modbus.func_code"].notna().mean(),

        "packet_count": WINDOW_SIZE,
    }

    windows.append(features)

win_df = pd.DataFrame(windows)
win_df.to_csv("../data/test/attack_windows.csv", index=False)
