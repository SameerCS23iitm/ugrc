import pandas as pd
import os

def split_csv(input_file, output_dir, chunk_size=1_000_000):
    os.makedirs(output_dir, exist_ok=True)

    for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
        out_file = os.path.join(output_dir, f"csc_{i}.csv")
        chunk.to_csv(out_file, index=False)
        print(f"Wrote {out_file}")

# usage
split_csv("../train/cscada_attack_ssw_analysis.csv", "../train/chunks", chunk_size=1_000_000)