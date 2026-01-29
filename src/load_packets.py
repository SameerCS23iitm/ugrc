import pandas as pd

def load_packets(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)

    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    # Debug once (you can remove later)
    print("Loaded columns:", df.columns.tolist())

    # ---- Handle time column safely ----
    if "frame.time_epoch" not in df.columns:
        raise ValueError(
            f"'frame.time_epoch' not found in {csv_path}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Convert time safely
    df["frame.time_epoch"] = pd.to_numeric(
        df["frame.time_epoch"], errors="coerce"
    )

    # Drop rows with invalid timestamps
    df = df.dropna(subset=["frame.time_epoch"])

    # ---- Convert numeric Modbus / packet fields ----
    numeric_cols = [
        "mbtcp.len",
        "mbtcp.unit_id",
        "modbus.func_code",
        "frame.len",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- Convert tcp.flags hex → int ----
    if "tcp.flags" in df.columns:
        df["tcp.flags"] = df["tcp.flags"].apply(
            lambda x: int(x, 16)
            if isinstance(x, str) and x.startswith("0x")
            else pd.NA
        )

    # Drop non-behavioral identifiers (do NOT use them later)
    df = df.drop(
        columns=["ip.src", "ip.dst", "mbtcp.trans_id"],
        errors="ignore",
    )

    # Final cleanup
    df = df.dropna().reset_index(drop=True)

    return df
