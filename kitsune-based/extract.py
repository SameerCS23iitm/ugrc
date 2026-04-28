import subprocess
from pathlib import Path

def extract(input_dir, output_csv):
    pcaps = sorted(p for p in Path(input_dir).glob("*.pcap") if p.name != "merged.pcap")
    merged_pcap = Path(input_dir) / "merged.pcap"

    print(f"Merging {len(pcaps)} pcap(s)...")
    subprocess.run(["mergecap", "-w", str(merged_pcap)] + [str(p) for p in pcaps], check=True)

    cmd = [
        "tshark", "-r", str(merged_pcap),
        "-T", "fields",
        # Frame
        "-e", "frame.time_epoch",
        "-e", "frame.len",
        # IP
        "-e", "ip.src",
        "-e", "ip.dst",
        # TCP
        "-e", "tcp.len",
        "-e", "tcp.analysis.retransmission",
        # MBAP header
        "-e", "mbtcp.trans_id",
        "-e", "mbtcp.unit_id",
        "-e", "mbtcp.len",
        # Modbus function & addressing
        "-e", "modbus.func_code",
        "-e", "modbus.exception_code",
        "-e", "modbus.reference_num",
        # Modbus counts
        "-e", "modbus.word_cnt",
        "-e", "modbus.byte_cnt",
        # Modbus values
        "-e", "modbus.regval_uint16",
        "-e", "modbus.data",
        # Modbus request/response linking
        "-e", "modbus.response_time",
        # Output format
        "-E", "header=y",
        "-E", "separator=,",
        "-E", "quote=d",
        "-E", "occurrence=a",
        "-E", "aggregator=|",
    ]

    print("Extracting fields...")
    with open(output_csv, "w") as f:
        subprocess.run(cmd, stdout=f, check=True)

    print("Saved:", output_csv)

extract("../data/benign/network-wide-pcap-capture/network-wide","../train/benign_nw_kit.csv")
extract("../data/attack/compromised-scada/substation-wide-capture", "../train/cscada_attack_ssw_kit.csv")
extract("../data/attack/external/network-wide", "../train/ext_attack_nw_kit.csv")

import os
os.system('notify-send "Python Script" "Execution complete!"')