import pandas as pd

WINDOW_SIZE = 0.2  # seconds

def aggregate_windows(df):
    # Sort by time
    df = df.sort_values("frame.time_epoch")

    # Create window index
    start_time = df["frame.time_epoch"].min()
    df["window_id"] = ((df["frame.time_epoch"] - start_time) // WINDOW_SIZE).astype(int)
    df["iat"] = df["frame.time_epoch"].diff().fillna(0)

    # Aggregate per window
    windows = df.groupby("window_id").agg(
        packet_count=("frame.len", "count"),
        mean_len=("frame.len", "mean"),
        std_len=("frame.len", "std"),
        mean_mb_len=("mbtcp.len", "mean"),
        unique_func=("modbus.func_code", "nunique"),
        unique_unit=("mbtcp.unit_id", "nunique"),
        mean_iat=("iat", "mean"),
        std_iat=("iat", "std"),
    ).reset_index(drop=True)


    # Replace NaNs (std with 1 packet)
    windows = windows.fillna(0)

    return windows
