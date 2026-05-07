# UGRC Modbus Anomaly Detection Project

## Project Overview

This project implements **machine learning-based anomaly detection** for substation networks using the **CIC Modbus Dataset 2023**. The goal is to detect intrusion attempts and anomalies in Industrial Control System (ICS) network traffic using the Modbus protocol.

### Key Objectives
- Detect malicious network traffic in substation environments
- Compare multiple anomaly detection approaches:
  - **KitNET**: Unsupervised ensemble of autoencoders for online anomaly detection
  - **LSTM-based models**: Including standard classification, autoencoders, and weighted variants
  - **Tree-based models**: XGBoost and Random Forest classifiers
- Evaluate performance across different attack scenarios (SCADA compromises, external attacks)
- Extract features from raw PCAP network captures and create labeled datasets

---

## Dataset Description

### CIC Modbus Dataset 2023
The dataset contains network traffic captures from a **simulated substation network** with Modbus protocol communication.

#### Dataset Structure
- **Benign Traffic**: Normal Modbus communication from legitimate devices
- **Attack Traffic**: 9 different types of attacks including:
  - Reconnaissance
  - Query flooding
  - Payload loading
  - Delay response
  - Parameter modification
  - False data injection
  - Frame stacking
  - Brute force write
  - Baseline replay

#### Attack Scenarios
1. **Compromised SCADA/HMI**: Attacks originating from internal devices
2. **External Network**: Attacks from outside the substation
3. **Compromised IED**: Attacks from industrial electronic devices

#### Network Architecture
```
IED1A (185.175.0.4) - Secure IED
IED1B (185.175.0.5) - Normal IED
IED4C (185.175.0.8) - Secure IED
SCADA HMI (185.175.0.2) - Secure, (185.175.0.3) - Normal
Central Agent (185.175.0.6)
Attacker (185.175.0.7)
```

---

## Project Structure

```
ugrc/
├── README.md                      # Original project info
├── requirements.txt               # Python dependencies
├── data/                          # Raw PCAP files and logs
│   ├── benign/                    # Benign network captures
│   ├── attack/                    # Attack scenarios
│   │   ├── compromised-scada/
│   │   ├── compromised-ied/
│   │   └── external/
│   └── combined/                  # Merged captures
│
├── train/                         # Processed & labeled datasets
│   ├── *.csv                      # Feature-extracted data
│   ├── *.npy                      # Numpy arrays for models
│   └── *.keras                    # Pre-trained models
│
├── analysis/                      # Data analysis & extraction
│   ├── 1.extract_csvs.ipynb       # Extract fields from PCAP → CSV
│   ├── 1.modbus_specific.ipynb    # Modbus protocol analysis
│   ├── common.ipynb               # Common utilities
│   ├── recon.ipynb                # Reconnaissance analysis
│   ├── wrapper.py                 # Helper functions
│   └── obs/                       # Observations/results
│
├── kitsune-based/                 # KitNET anomaly detector
│   ├── kitsune_new.py             # Main KitNET implementation
│   ├── kitnet_modbus_baseline.py  # Baseline evaluation
│   ├── extract.py                 # PCAP extraction for KitNET
│   ├── windowing.py               # Create time windows
│   ├── run_kitnet.sh              # Execution script
│   └── obs/                       # Results & observations
│
├── lstm-based/                    # LSTM-based models
│   ├── 3.labeling.py              # Label datasets with attack ground truth
│   ├── 4a.weighted_lstm.py        # Weighted LSTM classifier
│   ├── 4b.autoenc_mse.py          # MSE-based autoencoder
│   ├── 4c.lstm_autoencoder.py     # Sequence autoencoder
│   ├── 4d.baseline_lstm.py        # Basic LSTM baseline
│   ├── rforest.py                 # Random Forest classifier
│   ├── xgb.py                     # XGBoost classifier
│   ├── xgboost_classifier.py      # XGBoost alternative
│   └── obs/                       # Results & observations
│
├── KitNET-py/                     # KitNET library
│   ├── KitNET.py                  # Core anomaly detection
│   ├── corClust.py                # Correlation clustering
│   ├── dA.py                       # Denoising autoencoders
│   ├── utils.py                   # Utilities
│   └── README.md                  # KitNET documentation
│
└── observations/                  # Aggregated experimental results
```

---

## Setup & Installation

### Prerequisites
- Python 3.8+
- `tcpdump` or `tshark` for PCAP processing
- Virtual environment (recommended)

### Installation Steps

1. **Clone and navigate to the project**
   ```bash
   cd ugrc
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Linux/Mac
   # or: .venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Key dependencies:
   - `scapy==2.7.0`: Packet manipulation and analysis
   - `pandas`: Data manipulation
   - `numpy`: Numerical computing
   - `scikit-learn`: Machine learning utilities
   - `torch/tensorflow`: Deep learning frameworks
   - `xgboost`: Gradient boosting
   - `matplotlib/seaborn`: Visualization
   - `jupyter`: Interactive notebooks

4. **Ensure tools are available** (for PCAP processing)
   ```bash
   # Linux/Mac
   brew install wireshark  # includes tshark & mergecap
   apt-get install wireshark  # Debian/Ubuntu
   
   # Or manually install tshark and mergecap
   ```

---

## Workflow & Data Pipeline

### Stage 1: Extract Features from PCAP Files
Converts raw network captures into structured feature data.

**Notebooks/Scripts:**
- [`analysis/1.extract_csvs.ipynb`](analysis/1.extract_csvs.ipynb): Extract Modbus fields from PCAP
- [`kitsune-based/extract.py`](kitsune-based/extract.py): Extract for KitNET format

**Process:**
```
Raw PCAP files → Merge captures → tshark extraction → CSV with fields:
  - Frame metadata: timestamp, length, source IP, destination IP
  - Modbus fields: function code, unit ID, exception code
  - Values: register values, word counts
```

**Output:** CSV files in `train/` directory (e.g., `1s_benign_flows.csv`)

---

### Stage 2: Feature Engineering & Windowing
Aggregate packet-level data into time windows and extract statistical features.

**Key Scripts:**
- [`kitsune-based/windowing.py`](kitsune-based/windowing.py): Create 100ms/1s windows
- [`lstm-based/2.feature_extraction.ipynb`](lstm-based/2.feature_extraction.ipynb): Extract features

**Features Extracted:**
```
- packet_count: Number of packets in window
- total_bytes: Sum of packet sizes
- mean_packet_size: Average packet size
- std_packet_size: Standard deviation of packet sizes
- iat_mean: Mean inter-arrival time
- iat_std: Standard deviation of inter-arrival time
- unique_func_codes: Count of unique Modbus function codes
- exception_count: Count of Modbus exceptions
- ... and more protocol-specific features
```

**Granularity Options:**
- **100ms windows**: Finer temporal resolution, more data points
- **1s windows**: Coarser resolution, fewer samples

**Output:** Windowed CSV files (e.g., `labeled_1s_cscada.csv`)

---

### Stage 3: Label Data with Attack Ground Truth
Align feature windows with attack logs to create labeled datasets.

**Scripts:**
- [`lstm-based/3.labeling.py`](lstm-based/3.labeling.py): Label windows with attack metadata
- Uses attack CSV logs: `TargetIP`, `Attack`, `TransactionID`, `Timestamp`

**Output:** Labeled CSV with columns: `[features..., label, time_window]`
- `label=0`: Benign traffic
- `label=1`: Attack traffic

---

### Stage 4: Train & Evaluate Models

#### Option A: KitNET (Unsupervised)

**KitNET** is an ensemble of autoencoders trained ONLY on benign data to detect anomalies.

**Run Training & Validation:**
```bash
cd kitsune-based
chmod +x run_kitnet.sh

# Validate on SCADA attack scenario (tunes threshold)
./run_kitnet.sh cscada validate

# Test on external attack scenario
./run_kitnet.sh external test

# With custom hyperparameters
./run_kitnet.sh cscada validate --max-ae 32 --learning-rate 0.05
./run_kitnet.sh external test --max-ae 16 --threshold-quantile 0.99
```

**Configuration Options:**
- `--max-ae`: Maximum autoencoder size (default: 10)
- `--learning-rate`: Training rate (default: 0.001)
- `--threshold-quantile`: Anomaly threshold percentile (default: 0.95)
- `--force-retrain`: Retrain even if model exists

**Output:** Models saved to `obs/` with evaluation metrics (ROC-AUC, precision, recall)

---

#### Option B: LSTM-Based Models (Supervised/Unsupervised)

**Run Individual Models:**

1. **Baseline LSTM Classifier** (supervised)
   ```bash
   cd lstm-based
   python 4d.baseline_lstm.py
   ```
   - Trains on labeled benign/attack data
   - Outputs classification metrics

2. **Weighted LSTM** (supervised with class weighting)
   ```bash
   python 4a.weighted_lstm.py
   ```
   - Addresses class imbalance in attack data

3. **LSTM Autoencoder** (unsupervised)
   ```bash
   python 4c.lstm_autoencoder.py
   ```
   - Trains on benign sequences, detects anomalies by reconstruction error
   - Requires manual threshold tuning

4. **Autoencoder with MSE** (unsupervised)
   ```bash
   python 4b.autoenc_mse.py
   ```
   - Simpler autoencoder variant

5. **XGBoost Classifier** (supervised)
   ```bash
   python xgb.py
   ```
   - Tree-based classification

6. **Random Forest Classifier** (supervised)
   ```bash
   python rforest.py
   ```
   - Ensemble of decision trees

---

#### Option C: Run Analysis Notebooks

Explore data and test approaches interactively:

```bash
# Data extraction and exploration
jupyter notebook analysis/1.extract_csvs.ipynb

# Modbus protocol analysis
jupyter notebook analysis/1.modbus_specific.ipynb

# KitNET baseline evaluation
jupyter notebook kitsune-based/1.extract_csvs.ipynb

# Feature extraction for LSTM
jupyter notebook lstm-based/2.feature_extraction.ipynb

# Dataset splitting (80/20)
jupyter notebook lstm-based/split_20_80.ipynb
```

---

## Running the Complete Pipeline

### From Raw PCAP to Model Evaluation (KitNET)

```bash
# 1. Extract features from PCAP files
cd kitsune-based
python -c "from extract import extract; extract('../data/benign/', 'benign.csv')"

# 2. Create time windows
python -c "from windowing import create_windows; create_windows('benign.csv', 'benign_1s.csv', window_size=1)"

# 3. Train KitNET on benign data and evaluate
./run_kitnet.sh cscada validate
./run_kitnet.sh external test
```

### From Raw PCAP to Model Evaluation (LSTM)

```bash
# 1. Extract features (see notebooks)
jupyter notebook 2.feature_extraction.ipynb

# 2. Label with attacks
python 3.labeling.py

# 3. Train and evaluate
python 4c.lstm_autoencoder.py
python xgb.py
```

---

## Model Comparison & Evaluation

### Metrics
Models are evaluated using:
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve
- **Precision/Recall**: True positive rate and false positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: True/False positives and negatives
- **ROC & PR Curves**: Threshold analysis

### Key Differences

| Model | Type | Supervision | Speed | Memory | Explainability |
|-------|------|-------------|-------|--------|-----------------|
| KitNET | Ensemble Autoencoders | Unsupervised | Fast | Low | Low |
| LSTM-AE | Sequence Autoencoder | Unsupervised | Medium | Medium | Low |
| LSTM Classifier | RNN | Supervised | Medium | Medium | Low |
| XGBoost | Gradient Boosting | Supervised | Fast | Medium | High |
| Random Forest | Ensemble Trees | Supervised | Fast | Medium | High |

---

## Data Files

### Pre-processed Training Data (in `train/`)

| File | Description |
|------|-------------|
| `labeled_1s_benign.csv` | 1-second windowed benign traffic (training) |
| `labeled_1s_cscada.csv` | 1-second windowed SCADA attacks |
| `labeled_1s_external.csv` | 1-second windowed external attacks |
| `labeled_1s_cscada_val.csv` | Validation split for SCADA |
| `labeled_1s_external_val.csv` | Validation split for external |
| `X_1s.npy` | Feature matrix (numpy format) |
| `y_1s.npy` | Label vector (numpy format) |
| `lstm_1s.keras` | Pre-trained LSTM model |

---

## Key Concepts

### Modbus Protocol
Modbus is an industrial communication standard for reading/writing registers in PLCs and sensors. Key elements:
- **Function Codes**: Operations (read coils, read registers, write registers, etc.)
- **Unit IDs**: Device identifiers on the network
- **Transactions**: Request-response pairs identified by Transaction IDs

### Anomaly Detection Approaches

**Unsupervised (KitNET, Autoencoders):**
- Train on benign data only
- Learn "normal" patterns
- Detect deviations via reconstruction error or ensemble voting
- Advantages: No attack labels needed, finds novel attacks
- Disadvantages: Threshold tuning, potentially high false positives

**Supervised (LSTM, XGBoost, Random Forest):**
- Require labeled benign and attack data
- Learn to classify normal vs. anomalous
- Better performance on known attacks
- Advantages: Clear metrics, lower false positives
- Disadvantages: May miss novel attacks

---

## Common Issues & Troubleshooting

### Issue: Missing tshark or mergecap
**Solution:** Install Wireshark package
```bash
sudo apt-get install wireshark  # Linux
brew install wireshark  # macOS
```

### Issue: Insufficient memory for large datasets
**Solution:** 
- Use smaller windows (100ms instead of 1s)
- Process in batches
- Reduce autoencoder ensemble size (--max-ae)

### Issue: Model not converging
**Solution:**
- Adjust learning rate
- Try different window sizes
- Check class imbalance (use weighted variants)
- Increase training epochs

### Issue: Threshold tuning for KitNET
**Solution:** 
- Use validation set with `validate` mode
- Adjust `--threshold-quantile` (0.90-0.99 range)
- Analyze ROC curve in results

---

## References & Publications

- **KitNET Paper**: Mirsky, Y., Doitshman, T., Elovici, Y., & Shabtai, A. (2018). "Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection". NDSS 2018
- **CIC Modbus Dataset**: Available at [CIC Datasets](https://www.unb.ca/cic/datasets/index.html)
- **Modbus Protocol**: [Modbus IDA](http://www.modbus.org/)

---

## Authors & Contact

For questions about this project:
- Dataset inquiries: Kwasi Boakye-Boateng (kwasi.boakye-boateng@unb.ca)
- Project repository: UGRC (University of Guelph Research Collaboration)

---

## License

This project uses the CIC Modbus Dataset which is open source. Please refer to the dataset's license terms for usage restrictions.

---

## Summary: Quick Start

```bash
# 1. Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run KitNET (quickest method)
cd kitsune-based
./run_kitnet.sh cscada validate
./run_kitnet.sh external test

# 3. Or run LSTM models
cd ../lstm-based
python 4c.lstm_autoencoder.py

# 4. View results in observations/ directory
```

For detailed analysis, start with the Jupyter notebooks in each subdirectory.
