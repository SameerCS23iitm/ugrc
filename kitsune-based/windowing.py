import pandas as pd
import numpy as np
from scipy.stats import entropy

TIME_OFFSET = 10800

def entropy_feature(x):
    counts = x.value_counts()
    if len(counts) <= 1:
        return 0.0
    return entropy(counts)

def coeff_var(x):
    mean = x.mean()
    if mean == 0:
        return 0.0
    return x.std() / mean

def process_file(k, input_csv, output_csv):
    print(f"Processing {input_csv}...")

    cols = [
        "frame.time_epoch", "frame.len", "ip.src", "ip.dst",
        "tcp.len", "tcp.analysis.retransmission",
        "mbtcp.trans_id", "mbtcp.unit_id", "mbtcp.len",
        "modbus.func_code", "modbus.exception_code", "modbus.reference_num",
        "modbus.regval_uint16", "modbus.byte_cnt", "modbus.word_cnt",
        "modbus.response_time", "modbus.data",
    ]

    # Only cast columns that are guaranteed single-value (no pipe aggregation)
    dtypes = {
        "frame.len": "float32",
        "tcp.len":   "float32",
        "ip.src":    "category",
        "ip.dst":    "category",
    }

    # All other numeric columns may have pipe-separated values due to occurrence=a
    # These are stripped to first value and cast after loading
    pipe_numeric_cols = [
        "mbtcp.len", "mbtcp.unit_id", "mbtcp.trans_id",
        "modbus.func_code", "modbus.exception_code", "modbus.byte_cnt",
        "modbus.word_cnt", "modbus.regval_uint16", "modbus.reference_num",
        "modbus.response_time", "tcp.analysis.retransmission",
    ]

    # ── Pass 1: discover known_ips and observed_windows ────────────────────────
    print("Pass 1: scanning for IPs and time windows...")
    known_ips        = set()
    observed_windows = set()

    for chunk in pd.read_csv(input_csv,
                             usecols=["frame.time_epoch", "ip.src", "ip.dst"],
                             dtype={"ip.src": "category", "ip.dst": "category"},
                             chunksize=500_000):
        chunk = chunk.dropna(subset=["ip.src", "ip.dst"])
        chunk["time_window"] = ((chunk["frame.time_epoch"] + TIME_OFFSET) * k).astype(int)
        known_ips        |= set(chunk["ip.src"].astype(str).unique()) | set(chunk["ip.dst"].astype(str).unique())
        observed_windows |= set(chunk["time_window"].unique())

    # Drop multicast/non-unicast IPs
    known_ips        = sorted(ip for ip in known_ips if not ip.startswith("224."))
    observed_windows = pd.Index(sorted(observed_windows), name="time_window")
    print(f"IPs found: {known_ips}")
    print(f"Total time windows: {len(observed_windows)}")

    # ── Per-IP accumulators ────────────────────────────────────────────────────
    ip_accumulators = {ip: None for ip in known_ips}

    # ── Pass 2: chunked aggregation ────────────────────────────────────────────
    print("Pass 2: aggregating chunks...")
    chunk_idx = 0

    for chunk in pd.read_csv(input_csv, usecols=cols, dtype=dtypes, chunksize=500_000, low_memory=False):
        chunk_idx += 1
        print(f"  chunk {chunk_idx}...", end="\r")

        # ── Strip pipe-separated values and cast to float32 ───────────────────
        for col in pipe_numeric_cols:
            if col in chunk.columns:
                s = chunk[col]
                if s.dtype == object:
                    s = s.astype(str).str.split("|").str[0]
                chunk[col] = pd.to_numeric(s, errors="coerce").astype("float32")

        # ── Drop rows missing critical routing/TCP info ────────────────────────
        # These rows have no IP or TCP context and cannot be attributed to any IP
        chunk = chunk.dropna(subset=["ip.src", "ip.dst", "tcp.len"])

        # Drop multicast rows
        chunk = chunk[~chunk["ip.src"].astype(str).str.startswith("224.")]
        chunk = chunk[~chunk["ip.dst"].astype(str).str.startswith("224.")]

        # ── True zero fills ───────────────────────────────────────────────────
        # NaN means the event did not occur, 0 is semantically correct
        chunk["tcp.analysis.retransmission"] = chunk["tcp.analysis.retransmission"].fillna(0)
        chunk["modbus.response_time"]        = chunk["modbus.response_time"].fillna(0)
        chunk["modbus.exception_code"]       = chunk["modbus.exception_code"].fillna(0)

        # ── FC-conditional zero fills ─────────────────────────────────────────
        # NaN means FC does not use this field; 0 is safe since we aggregate
        # only over relevant FC subsets
        chunk["modbus.reference_num"]  = chunk["modbus.reference_num"].fillna(0)
        chunk["modbus.regval_uint16"]  = chunk["modbus.regval_uint16"].fillna(0)
        chunk["modbus.word_cnt"]       = chunk["modbus.word_cnt"].fillna(0)
        chunk["modbus.byte_cnt"]       = chunk["modbus.byte_cnt"].fillna(0)
        chunk["modbus.data"]           = chunk["modbus.data"].fillna("")

        # mbtcp.* left as NaN for non-Modbus packets intentionally —
        # derived columns below are computed only on modbus_mask rows

        # ── Derived columns on Modbus rows only ───────────────────────────────
        modbus_mask = chunk["modbus.func_code"].notna()
        chunk["is_stacked"]   = 0.0
        chunk["stack_excess"] = 0.0
        chunk["len_mismatch"] = 0.0
        chunk["is_mismatch"]  = 0.0

        chunk.loc[modbus_mask, "is_stacked"]   = (
            chunk.loc[modbus_mask, "tcp.len"] > (chunk.loc[modbus_mask, "mbtcp.len"] + 6)
        ).astype(float)
        chunk.loc[modbus_mask, "stack_excess"] = (
            chunk.loc[modbus_mask, "tcp.len"] - (chunk.loc[modbus_mask, "mbtcp.len"] + 6)
        ).clip(lower=0)
        chunk.loc[modbus_mask, "len_mismatch"] = (
            chunk.loc[modbus_mask, "mbtcp.len"] - chunk.loc[modbus_mask, "modbus.byte_cnt"]
        )
        chunk.loc[modbus_mask, "is_mismatch"]  = (
            chunk.loc[modbus_mask, "len_mismatch"].abs() > 2
        ).astype(float)

        chunk["is_duplicate"] = chunk.duplicated(
            subset=["ip.src", "ip.dst", "mbtcp.trans_id", "modbus.data"], keep=False
        ).astype(float)

        # ── Timestamp & windowing ─────────────────────────────────────────────
        chunk["timestamp"]   = chunk["frame.time_epoch"] + TIME_OFFSET
        chunk = chunk.sort_values("timestamp")
        chunk["time_window"] = (chunk["timestamp"] * k).astype(int)

        # ── IAT within (time_window, src, dst) ───────────────────────────────
        chunk["iat"] = (
            chunk.groupby(["time_window", "ip.src", "ip.dst"])["timestamp"]
            .diff()
            .fillna(0)
        )

        # ── Per-IP aggregation ────────────────────────────────────────────────
        for ip in known_ips:
            tx = chunk[chunk["ip.src"].astype(str) == ip]
            rx = chunk[chunk["ip.dst"].astype(str) == ip]

            if len(tx) == 0 and len(rx) == 0:
                continue

            tx_agg = tx.groupby("time_window").agg(
                # Volume
                tx_packet_count       = ("frame.len",                   "count"),
                tx_total_bytes        = ("frame.len",                   "sum"),
                tx_mean_pkt_size      = ("frame.len",                   "mean"),
                tx_std_pkt_size       = ("frame.len",                   "std"),
                # IAT
                tx_iat_mean           = ("iat",                         "mean"),
                tx_iat_std            = ("iat",                         "std"),
                tx_iat_cv             = ("iat",                         coeff_var),
                # Destinations & slaves
                tx_unique_dst         = ("ip.dst",                      "nunique"),
                tx_unique_unit_ids    = ("mbtcp.unit_id",               "nunique"),
                # Function codes
                tx_unique_fc          = ("modbus.func_code",            "nunique"),
                tx_func_entropy       = ("modbus.func_code",            entropy_feature),
                tx_read_count         = ("modbus.func_code",            lambda x: x.isin([3, 4]).sum()),
                tx_write_count        = ("modbus.func_code",            lambda x: x.isin([5, 6, 15, 16]).sum()),
                tx_fc3                = ("modbus.func_code",            lambda x: (x == 3).sum()),
                tx_fc4                = ("modbus.func_code",            lambda x: (x == 4).sum()),
                tx_fc5                = ("modbus.func_code",            lambda x: (x == 5).sum()),
                tx_fc6                = ("modbus.func_code",            lambda x: (x == 6).sum()),
                tx_fc15               = ("modbus.func_code",            lambda x: (x == 15).sum()),
                tx_fc16               = ("modbus.func_code",            lambda x: (x == 16).sum()),
                # Exceptions
                tx_exception_count    = ("modbus.exception_code",       lambda x: x.notna().sum()),
                # Register addressing
                tx_unique_regs        = ("modbus.reference_num",        "nunique"),
                tx_register_std       = ("modbus.reference_num",        "std"),
                tx_register_entropy   = ("modbus.reference_num",        entropy_feature),
                # Register values
                tx_regval_mean        = ("modbus.regval_uint16",        "mean"),
                tx_regval_std         = ("modbus.regval_uint16",        "std"),
                # Payload sizes
                tx_mean_mbap_len      = ("mbtcp.len",                   "mean"),
                tx_byte_cnt_mean      = ("modbus.byte_cnt",             "mean"),
                tx_word_cnt_mean      = ("modbus.word_cnt",             "mean"),
                # Response timing
                tx_resp_time_mean     = ("modbus.response_time",        "mean"),
                tx_resp_time_std      = ("modbus.response_time",        "std"),
                # Retransmissions
                tx_retransmission_sum = ("tcp.analysis.retransmission", "sum"),
                # Stacking
                tx_stacked_count      = ("is_stacked",                  "sum"),
                tx_mean_stack_excess  = ("stack_excess",                "mean"),
                # Length mismatch
                tx_len_mismatch_mean  = ("len_mismatch",                "mean"),
                tx_len_mismatch_count = ("is_mismatch",                 "sum"),
                # Duplicate / replay
                tx_duplicate_count    = ("is_duplicate",                "sum"),
                # Transaction IDs
                tx_unique_tids        = ("mbtcp.trans_id",              "nunique"),
            )

            rx_agg = rx.groupby("time_window").agg(
                rx_packet_count       = ("frame.len",                   "count"),
                rx_total_bytes        = ("frame.len",                   "sum"),
                rx_unique_src         = ("ip.src",                      "nunique"),
                rx_unique_unit_ids    = ("mbtcp.unit_id",               "nunique"),
                rx_retransmission_sum = ("tcp.analysis.retransmission", "sum"),
                rx_resp_time_mean     = ("modbus.response_time",        "mean"),
            )

            ip_agg = tx_agg.join(rx_agg, how="outer")
            ip_agg = ip_agg.reindex(observed_windows, fill_value=0).fillna(0)

            if ip_accumulators[ip] is None:
                ip_accumulators[ip] = ip_agg
            else:
                ip_accumulators[ip] = ip_accumulators[ip].add(ip_agg, fill_value=0)

    # ── Finalize: derived ratios and join all IPs ──────────────────────────────
    print("\nFinalizing...")
    result = pd.DataFrame(index=observed_windows)

    # for ip in known_ips:
    #     ip_df = ip_accumulators[ip]
    #     if ip_df is None:
    #         continue

    #     pkt = ip_df["tx_packet_count"].to_numpy(dtype=float)

    #     def safe_div(num_col):
    #         num = ip_df[num_col].to_numpy(dtype=float)
    #         return np.divide(num, pkt, out=np.zeros_like(pkt), where=pkt > 0)

    #     ip_df["tx_write_ratio"]     = safe_div("tx_write_count")
    #     ip_df["tx_read_ratio"]      = safe_div("tx_read_count")
    #     ip_df["tx_exception_ratio"] = safe_div("tx_exception_count")
    #     ip_df["tx_dup_tids"]        = ip_df["tx_packet_count"] - ip_df["tx_unique_tids"]
    #     ip_df["tx_tid_ratio"]       = safe_div("tx_unique_tids")
    #     ip_df["tx_regval_delta"]    = ip_df["tx_regval_mean"].diff().fillna(0)

    #     safe_ip = ip.replace(".", "_")
    #     ip_df   = ip_df.add_prefix(f"{safe_ip}__")
    #     result  = result.join(ip_df, how="left")

    # result = result.fillna(0).reset_index()

    all_ip_dfs = []
    for ip in known_ips:
        ip_df = ip_accumulators[ip]
        if ip_df is None:
            continue
        pkt = ip_df["tx_packet_count"].to_numpy(dtype=float)
        def safe_div(num_col):
            num = ip_df[num_col].to_numpy(dtype=float)
            return np.divide(num, pkt, out=np.zeros_like(pkt), where=pkt > 0)
        ip_df["tx_write_ratio"]     = safe_div("tx_write_count")
        ip_df["tx_read_ratio"]      = safe_div("tx_read_count")
        ip_df["tx_exception_ratio"] = safe_div("tx_exception_count")
        ip_df["tx_dup_tids"]        = ip_df["tx_packet_count"] - ip_df["tx_unique_tids"]
        ip_df["tx_tid_ratio"]       = safe_div("tx_unique_tids")
        ip_df["tx_regval_delta"]    = ip_df["tx_regval_mean"].diff().fillna(0)
        safe_ip = ip.replace(".", "_")
        ip_df   = ip_df.add_prefix(f"{safe_ip}__")
        all_ip_dfs.append(ip_df)

    result = pd.concat(all_ip_dfs, axis=1).reindex(observed_windows).fillna(0).reset_index().copy()
    result.to_csv(output_csv, index=False)
    result.to_csv(output_csv, index=False)
    print(f"Saved → {output_csv}")

process_file(1, "../train/ext_attack_nw_kit.csv", "../train/external_flows_kit.csv")
process_file(1, "../train/benign_nw_analysis.csv", "../train/benign_flows_kit.csv")
process_file(1, "../train/cscada_attack_ssw_kit.csv", "../train/cscada_flows_kit.csv")

import os
os.system('notify-send "Python Script" "Execution complete!"')