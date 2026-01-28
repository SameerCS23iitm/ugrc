import pandas as pd

df = pd.read_csv("../data/attack_csv/attack_ied1a.csv", low_memory=False)

# Remove duplicated header rows if any
df = df[df["frame.time_epoch"] != "frame.time_epoch"]

cols = [
    "frame.time_epoch",
    "tcp.len",
    "frame.len",
    "tcp.flags",
    "mbtcp.trans_id",
    "mbtcp.len",
    "mbtcp.unit_id",
    "modbus.func_code",
]
df = df[cols]

def parse_tcp_flags(x):
    if isinstance(x, str) and x.startswith("0x"):
        try:
            return int(x, 16)
        except ValueError:
            return None
    return x

df["tcp.flags"] = df["tcp.flags"].apply(parse_tcp_flags)

df = df.apply(pd.to_numeric, errors="coerce")

df = df.dropna(subset=["frame.time_epoch"])

df = df.sort_values("frame.time_epoch")
df["iat"] = df["frame.time_epoch"].diff().fillna(0)

df.to_csv("../data/test/attack_ft.csv", index=False)
