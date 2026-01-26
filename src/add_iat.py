import pandas as pd

df = pd.read_csv("../data/benign_csv/veth4edc015-normal-10.csv")
df["iat"] = df["frame.time_epoch"].diff().fillna(0)
df.to_csv("../data/train/veth4edc015-normal-10_ft.csv", index=False)

