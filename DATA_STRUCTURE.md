# X_train Data Structure - Complete Breakdown

## TL;DR
**X_train shape: `(532384, 20, 8)`**
- **532,384 sequences** of Modbus network traffic
- Each sequence = **20 consecutive time windows** (1 second each)
- Each window = **8 statistical features** computed from raw packets

---

## What Does Each Vector Contain?

### The 8 Features (per 1-second window):

| Feature | Description | Source |
|---------|-------------|--------|
| `packet_count` | Number of packets in the 1-second window | Raw pcap data |
| `total_bytes` | Total bytes transmitted in the window | Raw pcap data |
| `mean_packet_size` | Average packet size (bytes) | Computed from packet sizes |
| `std_packet_size` | Standard deviation of packet sizes | Computed from packet sizes |
| `iat_mean` | **Inter-Arrival Time (IAT) mean** - avg time between consecutive packets | Computed from packet timestamps |
| `iat_std` | Inter-Arrival Time std dev | Computed from packet timestamps |
| `unique_func_codes` | Number of unique **Modbus function codes** used | Modbus protocol extraction |
| `exception_count` | Number of **Modbus exceptions** (errors) | Modbus protocol extraction |

### Why These Features?
- **Packet statistics** (count, bytes, sizes) → Network behavior patterns
- **IAT metrics** (inter-arrival time) → Detects timing anomalies (attacks often change communication patterns)
- **Modbus-specific** (function codes, exceptions) → Protocol-level anomalies in industrial control systems

---

## How X_train is Built (Step-by-Step)

### Step 1: Extract Raw Packets from PCAPs
```
Raw PCAP files → tshark extraction → CSV with columns:
[timestamp, frame_length, ip.src, ip.dst, modbus.func_code, modbus.exception_code]
```
**Source**: `src/extract_csvs.ipynb`

### Step 2: Aggregate into Time Windows
```
Per IP pair (src, dst), compute 8 features per 1-second window:

Time      | IP Pair (src, dst) | packet_count | total_bytes | ... | exception_count
0.000s    | 192.168.1.1:1.2    | 5            | 234         | ... | 0
1.000s    | 192.168.1.1:1.2    | 3            | 120         | ... | 1
2.000s    | 192.168.1.1:1.2    | 7            | 456         | ... | 0
...
```
**Source**: Raw pcap extraction (tshark) → labeled_1s_*.csv files

### Step 3: Label Each Window (0=benign, 1=attack)
```
- Benign traffic: Label = 0
- Compromised-SCADA attacks: Label based on attack log timestamps (±10 second tolerance)
- External attacks: Label based on attacker IP (185.175.0.7)
```
**Source**: `src/labeling.ipynb`

### Step 4: Create Sequential Samples (20-Timestep Sequences)

For each IP pair, create sliding windows of 20 consecutive 1-second windows:

```
Sequence 1: Windows [0-19]   → Label: window[20]
Sequence 2: Windows [1-20]   → Label: window[21]
Sequence 3: Windows [2-21]   → Label: window[22]
...
```

**Each sequence shape**: `(20, 8)` = 20 time steps × 8 features

```python
def build_sequences(df, seq_len=20):
    for (src, dst), group in df.groupby(["ip.src", "ip.dst"]):
        features = group[FEATURE_COLS].values  # shape: (N, 8)
        
        for i in range(len(group) - seq_len):
            X.append(features[i:i+seq_len])    # Append (20, 8) sequence
            y.append(labels[i+seq_len])        # Label of the NEXT window
    
    return np.array(X), np.array(y)
```

### Step 5: Normalize and Split

```
All 8 features normalized using MinMaxScaler (0-1 range)
↓
80% train / 20% test split
↓
Final: X_train (532384, 20, 8) | y_train (532384,)
Final: X_test  (133096, 20, 8) | y_test  (133096,)
```

---

## What Each Row in X_train Represents

**One row in X_train** = A 20-second observation window of Modbus traffic

Example structure (schematic):
```
X_train[0] = [
    [5, 234, 46.8, 12.3, 0.15, 0.08, 2, 0],    # Second 0 (timestamp T)
    [3, 120, 40.0,  8.5, 0.18, 0.12, 2, 0],    # Second 1 (timestamp T+1)
    [7, 456, 65.1, 15.2, 0.12, 0.06, 3, 1],    # Second 2 (timestamp T+2)
    ... (17 more rows)
    [4, 180, 45.0, 10.1, 0.20, 0.10, 2, 0],    # Second 19 (timestamp T+19)
]
y_train[0] = 1  # The label of Second 20 (timestamp T+20) - is it attack?
```

Each row represents **20 consecutive seconds** of aggregated network traffic statistics for a specific source-destination IP pair.

---

## Class Distribution

From your diagnostic run:
- **Benign sequences**: ~95% (508k samples)
- **Attack sequences**: ~5% (24k samples)

This is why you needed class weighting! The model sees mostly benign data.

---

## Summary Table

| Dimension | Meaning |
|-----------|---------|
| **Axis 0** (532384) | Number of sequences (observations) |
| **Axis 1** (20) | Time steps (consecutive 1-second windows) |
| **Axis 2** (8) | Statistical features from network/Modbus traffic |

---

## Related Files

| File | Purpose |
|------|---------|
| `src/extract_csvs.ipynb` | Extract raw packets from PCAPs using tshark |
| `src/labeling.ipynb` | Create sequences, label windows, normalize features |
| `train/X_train_1s.npy` | Your actual X_train data (532384, 20, 8) |
| `train/y_train_1s.npy` | Labels for X_train (532384,) |
| `train/lstm_config.json` | Saved optimal threshold & class weights for inference |

