"""
=============================================================
  Custom Modbus TCP Flow Feature Extractor
  for ML-based Anomaly Detection
=============================================================
  Input  : one or more PCAP / PCAPNG files
  Output : CSV with one row per Modbus TCP flow (~60 features)

  Usage:
      python modbus_parser.py --pcap traffic.pcap --out features.csv
      python modbus_parser.py --pcap pcaps/  --out features.csv --label benign
      python modbus_parser.py --pcap a.pcap b.pcap --out out.csv --label dos_attack

  Optional flags:
      --label <str>   Label string attached to every row  (e.g. "benign")
      --port  <int>   Modbus TCP port (default: 502)

  Requirements:
      pip install dpkt
=============================================================
"""

import argparse
import csv
import glob
import math
import os
import struct
import sys
from collections import defaultdict

import dpkt


# ══════════════════════════════════════════════════════════════════════════════
#  MODBUS CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

FC_NAMES = {
    1:  "Read_Coils",
    2:  "Read_Discrete_Inputs",
    3:  "Read_Holding_Registers",
    4:  "Read_Input_Registers",
    5:  "Write_Single_Coil",
    6:  "Write_Single_Register",
    7:  "Read_Exception_Status",
    8:  "Diagnostics",
    11: "Get_Comm_Event_Counter",
    12: "Get_Comm_Event_Log",
    15: "Write_Multiple_Coils",
    16: "Write_Multiple_Registers",
    17: "Report_Server_ID",
    20: "Read_File_Record",
    21: "Write_File_Record",
    22: "Mask_Write_Register",
    23: "Read_Write_Multiple_Registers",
    24: "Read_FIFO_Queue",
    43: "Encapsulated_Interface_Transport",
}

READ_FCS  = {1, 2, 3, 4, 7, 11, 12, 17, 20, 24}
WRITE_FCS = {5, 6, 15, 16, 21, 22, 23}

EXCEPTION_OFFSET  = 0x80
MODBUS_HEADER_LEN = 7      # TxID(2) + ProtoID(2) + Length(2) + UnitID(1)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _stats(values):
    n = len(values)
    if n == 0:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0
    mn  = min(values)
    mx  = max(values)
    avg = sum(values) / n
    var = sum((x - avg) ** 2 for x in values) / n if n > 1 else 0.0
    return n, mn, mx, avg, math.sqrt(var), var


def _iats_us(ts_list):
    if len(ts_list) < 2:
        return []
    s = sorted(ts_list)
    return [(s[i + 1] - s[i]) * 1e6 for i in range(len(s) - 1)]


def _ip_str(addr_bytes):
    return ".".join(str(b) for b in addr_bytes)


# ══════════════════════════════════════════════════════════════════════════════
#  MODBUS PDU PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_modbus_payload(payload: bytes) -> list:
    messages = []
    offset   = 0

    while offset + MODBUS_HEADER_LEN <= len(payload):
        tx_id    = struct.unpack_from(">H", payload, offset)[0]
        proto_id = struct.unpack_from(">H", payload, offset + 2)[0]
        length   = struct.unpack_from(">H", payload, offset + 4)[0]
        unit_id  = payload[offset + 6]

        if proto_id != 0 or length < 2:
            break

        pdu_start = offset + MODBUS_HEADER_LEN
        pdu_end   = offset + MODBUS_HEADER_LEN + (length - 1)

        if pdu_end > len(payload):
            break

        pdu = payload[pdu_start:pdu_end]
        if not pdu:
            break

        fc           = pdu[0]
        pdu_data     = pdu[1:]
        is_exception = fc >= EXCEPTION_OFFSET
        real_fc      = (fc - EXCEPTION_OFFSET) if is_exception else fc
        exc_code     = pdu_data[0] if (is_exception and pdu_data) else None

        start_addr = None
        quantity   = None
        if not is_exception and real_fc in (1, 2, 3, 4, 5, 6, 15, 16):
            if len(pdu_data) >= 2:
                start_addr = struct.unpack_from(">H", pdu_data, 0)[0]
            if real_fc in (1, 2, 3, 4, 15, 16) and len(pdu_data) >= 4:
                quantity = struct.unpack_from(">H", pdu_data, 2)[0]
            elif real_fc in (5, 6):
                quantity = 1

        messages.append({
            "tx_id":        tx_id,
            "unit_id":      unit_id,
            "fc":           real_fc,
            "is_exception": is_exception,
            "exc_code":     exc_code,
            "pdu_len":      len(pdu),
            "adu_len":      MODBUS_HEADER_LEN + (length - 1),
            "start_addr":   start_addr,
            "quantity":     quantity,
        })

        offset = pdu_end

    return messages


# ══════════════════════════════════════════════════════════════════════════════
#  FLOW ACCUMULATOR
# ══════════════════════════════════════════════════════════════════════════════

class ModbusFlow:
    def __init__(self, key):
        self.key = key
        self.first_ts = self.last_ts = None
        self.ts_fwd, self.ts_bwd = [], []
        self.msgs_fwd, self.msgs_bwd = [], []
        self.open_req = {}
        self.rtts     = []

    def add(self, ts: float, is_fwd: bool, payload: bytes):
        if self.first_ts is None:
            self.first_ts = ts
        self.last_ts = ts
        (self.ts_fwd if is_fwd else self.ts_bwd).append(ts)

        for m in parse_modbus_payload(payload):
            m["ts"] = ts
            if is_fwd:
                self.msgs_fwd.append(m)
                self.open_req[m["tx_id"]] = ts
            else:
                self.msgs_bwd.append(m)
                req_ts = self.open_req.pop(m["tx_id"], None)
                if req_ts is not None:
                    self.rtts.append((ts - req_ts) * 1e6)

    def features(self, label="") -> dict:
        all_msgs = self.msgs_fwd + self.msgs_bwd
        duration  = (self.last_ts - self.first_ts) if self.first_ts else 0.0

        fc_cnt = defaultdict(int)
        for m in all_msgs:
            fc_cnt[m["fc"]] += 1

        exc_fwd   = [m for m in self.msgs_fwd if m["is_exception"]]
        exc_bwd   = [m for m in self.msgs_bwd if m["is_exception"]]
        total_exc = len(exc_fwd) + len(exc_bwd)
        exc_rate  = total_exc / len(all_msgs) if all_msgs else 0.0
        uniq_exc  = len({m["exc_code"] for m in all_msgs if m["is_exception"] and m["exc_code"] is not None})

        _, pdu_min,  pdu_max,  pdu_mean,  pdu_std,  pdu_var = _stats([m["pdu_len"] for m in all_msgs])
        _, fpdu_min, fpdu_max, fpdu_mean, fpdu_std, _        = _stats([m["pdu_len"] for m in self.msgs_fwd])
        _, bpdu_min, bpdu_max, bpdu_mean, bpdu_std, _        = _stats([m["pdu_len"] for m in self.msgs_bwd])
        total_bytes = sum(m["adu_len"] for m in all_msgs)

        addrs = [m["start_addr"] for m in all_msgs if m["start_addr"] is not None]
        qtys  = [m["quantity"]   for m in all_msgs if m["quantity"]   is not None]
        _, addr_min, addr_max, addr_mean, addr_std, _ = _stats(addrs)
        addr_range = (addr_max - addr_min) if addrs else 0.0
        _, qty_min, qty_max, qty_mean, qty_std, _     = _stats(qtys)

        tx_ids  = [m["tx_id"] for m in self.msgs_fwd]
        tx_gaps = [abs(tx_ids[i] - tx_ids[i-1]) for i in range(1, len(tx_ids))]
        _, _, tx_gap_max, tx_gap_mean, tx_gap_std, _ = _stats(tx_gaps)
        tx_reuse = len(tx_ids) - len(set(tx_ids))

        n_all   = len(all_msgs)
        n_req   = len(self.msgs_fwd)
        n_resp  = len(self.msgs_bwd)
        r_ratio = sum(fc_cnt[f] for f in READ_FCS)  / n_all if n_all else 0
        w_ratio = sum(fc_cnt[f] for f in WRITE_FCS) / n_all if n_all else 0
        pair_r  = len(self.rtts) / n_req if n_req else 0.0
        _, rtt_min, rtt_max, rtt_mean, rtt_std, _ = _stats(self.rtts)

        iat_all = _iats_us(self.ts_fwd + self.ts_bwd)
        iat_fwd = _iats_us(self.ts_fwd)
        iat_bwd = _iats_us(self.ts_bwd)
        _, iat_min,  iat_max,  iat_mean,  iat_std,  _ = _stats(iat_all)
        _, fiat_min, fiat_max, fiat_mean, fiat_std, _  = _stats(iat_fwd)
        _, biat_min, biat_max, biat_mean, biat_std, _  = _stats(iat_bwd)

        bps = total_bytes / duration if duration > 0 else 0.0
        pps = n_all        / duration if duration > 0 else 0.0

        src_ip, src_port, dst_ip, dst_port = self.key

        return {
            "src_ip":   src_ip,   "src_port": src_port,
            "dst_ip":   dst_ip,   "dst_port": dst_port,
            "flow_duration_s":   round(duration, 6),
            "flow_start_ts":     round(self.first_ts, 6) if self.first_ts else 0,
            "total_modbus_msgs": n_all,
            "fwd_msgs":          n_req,
            "bwd_msgs":          n_resp,
            "fwd_bwd_ratio":     round(n_req / n_resp, 4) if n_resp else 0,
            "total_modbus_bytes":  total_bytes,
            "modbus_bytes_per_s":  round(bps, 4),
            "modbus_pkts_per_s":   round(pps, 4),
            "pdu_len_min":    round(pdu_min, 2),   "pdu_len_max":  round(pdu_max, 2),
            "pdu_len_mean":   round(pdu_mean, 4),  "pdu_len_std":  round(pdu_std, 4),
            "pdu_len_var":    round(pdu_var, 4),
            "fwd_pdu_len_min":  round(fpdu_min, 2),  "fwd_pdu_len_max":  round(fpdu_max, 2),
            "fwd_pdu_len_mean": round(fpdu_mean, 4), "fwd_pdu_len_std":  round(fpdu_std, 4),
            "bwd_pdu_len_min":  round(bpdu_min, 2),  "bwd_pdu_len_max":  round(bpdu_max, 2),
            "bwd_pdu_len_mean": round(bpdu_mean, 4), "bwd_pdu_len_std":  round(bpdu_std, 4),
            "fc_read_coils_cnt":               fc_cnt.get(1,  0),
            "fc_read_discrete_inputs_cnt":     fc_cnt.get(2,  0),
            "fc_read_holding_registers_cnt":   fc_cnt.get(3,  0),
            "fc_read_input_registers_cnt":     fc_cnt.get(4,  0),
            "fc_write_single_coil_cnt":        fc_cnt.get(5,  0),
            "fc_write_single_register_cnt":    fc_cnt.get(6,  0),
            "fc_write_multiple_coils_cnt":     fc_cnt.get(15, 0),
            "fc_write_multiple_registers_cnt": fc_cnt.get(16, 0),
            "fc_diagnostics_cnt":              fc_cnt.get(8,  0),
            "fc_other_cnt": sum(v for k, v in fc_cnt.items() if k not in (1,2,3,4,5,6,8,15,16)),
            "unique_fcs_used":   len(fc_cnt),
            "read_msg_ratio":    round(r_ratio, 4),
            "write_msg_ratio":   round(w_ratio, 4),
            "total_exception_msgs":   total_exc,
            "exception_rate":         round(exc_rate, 6),
            "fwd_exception_cnt":      len(exc_fwd),
            "bwd_exception_cnt":      len(exc_bwd),
            "unique_exception_codes": uniq_exc,
            "addr_min":   round(addr_min, 2),   "addr_max":   round(addr_max, 2),
            "addr_mean":  round(addr_mean, 4),  "addr_std":   round(addr_std, 4),
            "addr_range": round(addr_range, 2),
            "qty_min":    round(qty_min, 2),    "qty_max":    round(qty_max, 2),
            "qty_mean":   round(qty_mean, 4),   "qty_std":    round(qty_std, 4),
            "tx_gap_mean":    round(tx_gap_mean, 4),
            "tx_gap_max":     round(tx_gap_max, 2),
            "tx_gap_std":     round(tx_gap_std, 4),
            "tx_reuse_count": tx_reuse,
            "request_response_pairing_rate": round(pair_r, 6),
            "rtt_mean_us": round(rtt_mean, 2), "rtt_std_us": round(rtt_std, 2),
            "rtt_max_us":  round(rtt_max, 2),
            "unmatched_requests": len(self.open_req),
            "iat_mean_us":     round(iat_mean, 2),  "iat_std_us":     round(iat_std, 2),
            "iat_min_us":      round(iat_min, 2),   "iat_max_us":     round(iat_max, 2),
            "fwd_iat_mean_us": round(fiat_mean, 2), "fwd_iat_std_us": round(fiat_std, 2),
            "bwd_iat_mean_us": round(biat_mean, 2), "bwd_iat_std_us": round(biat_std, 2),
            "unique_unit_ids": len({m["unit_id"] for m in all_msgs}),
            "label": label,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  PCAP PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

def process_pcap(path: str, modbus_port: int = 502, label: str = "") -> list:
    print(f"  [*] Reading {path} ...", end=" ", flush=True)
    flows = {}

    try:
        with open(path, "rb") as f:
            try:
                reader = dpkt.pcap.Reader(f)
            except ValueError:
                f.seek(0)
                reader = dpkt.pcapng.Reader(f)

            for ts, buf in reader:
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                except Exception:
                    continue
                if not isinstance(eth.data, dpkt.ip.IP):
                    continue
                ip  = eth.data
                if not isinstance(ip.data, dpkt.tcp.TCP):
                    continue
                tcp  = ip.data
                data = bytes(tcp.data)
                if not data or len(data) < MODBUS_HEADER_LEN:
                    continue
                if tcp.dport != modbus_port and tcp.sport != modbus_port:
                    continue

                src_ip = _ip_str(ip.src)
                dst_ip = _ip_str(ip.dst)

                if tcp.dport == modbus_port:
                    key    = (src_ip, tcp.sport, dst_ip, tcp.dport)
                    is_fwd = True
                else:
                    key    = (dst_ip, tcp.dport, src_ip, tcp.sport)
                    is_fwd = False

                if key not in flows:
                    flows[key] = ModbusFlow(key)
                flows[key].add(ts, is_fwd, data)

    except FileNotFoundError:
        print(f"\n  [!] File not found: {path}")
        return []
    except Exception as e:
        print(f"\n  [!] Error: {e}")
        return []

    result = [f.features(label=label) for f in flows.values()]
    print(f"-> {len(result)} flows extracted.")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def resolve_paths(inputs):
    paths = []
    for inp in inputs:
        if os.path.isdir(inp):
            paths += glob.glob(os.path.join(inp, "*.pcap"))
            paths += glob.glob(os.path.join(inp, "*.pcapng"))
        elif os.path.isfile(inp):
            paths.append(inp)
        else:
            found = glob.glob(inp)
            if found:
                paths.extend(found)
            else:
                print(f"[!] Not found: {inp}")
    return sorted(set(paths))


def main():
    ap = argparse.ArgumentParser(
        description="Modbus TCP PCAP -> ML feature CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python modbus_parser.py --pcap traffic.pcap --out features.csv
  python modbus_parser.py --pcap pcaps/ --out features.csv --label benign
  python modbus_parser.py --pcap a.pcap b.pcap --out out.csv --label dos_attack
        """,
    )
    ap.add_argument("--pcap",  nargs="+", required=True)
    ap.add_argument("--out",   default="modbus_features.csv")
    ap.add_argument("--label", default="")
    ap.add_argument("--port",  type=int, default=502)
    args = ap.parse_args()

    files = resolve_paths(args.pcap)
    if not files:
        print("[!] No PCAP files found.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Modbus Parser | port={args.port} | label='{args.label}'")
    print(f"{'='*60}")
    print(f"  Files: {len(files)}\n")

    all_rows = []
    for pf in files:
        all_rows.extend(process_pcap(pf, args.port, args.label))

    if not all_rows:
        print("\n[!] No Modbus flows detected.")
        sys.exit(1)

    cols = list(all_rows[0].keys())
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)

    print(f"\n{'='*60}")
    print(f"  Done! {len(all_rows)} flows -> {os.path.abspath(args.out)}")
    print(f"  Feature columns : {len(cols)}")
    print(f"{'='*60}\n")
    fc3 = sum(r["fc_read_holding_registers_cnt"] for r in all_rows)
    exc = sum(r["total_exception_msgs"]           for r in all_rows)
    print(f"  FC3 (Read Holding Registers) msgs : {fc3}")
    print(f"  Total exception responses         : {exc}\n")


if __name__ == "__main__":
    main()
