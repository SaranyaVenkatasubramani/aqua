# AquaSentinel

**Catching water leaks early from sensor readings.** Team Quantiq, IBM Z Datathon 2026.

AquaSentinel learns what a normal day looks like for a water network (separate weekday and weekend profiles), turns the last 1, 6 and 24 hours of pressure and flow into change signals, and gives a leak probability for every 5-minute reading. An alert fires when the smoothed score stays high for an hour, and the dashboard points to the sensors with the largest pressure drop. The pipeline was run on an IBM LinuxONE (s390x) virtual server.


<img width="3200" height="1800" alt="architecture" src="https://github.com/user-attachments/assets/1c01eb7e-03b7-4ebd-9e1e-b2ca1877faa1" />

## Results (2019 test year, never seen in training)

Trained on 2018, tested on all 105,120 readings of 2019.

| Metric | Value |
|---|---|
| Accuracy | 96.05% |
| Precision | 96.60% |
| Recall (leaks caught) | 98.60% |
| F1-score | 97.59% |
| ROC-AUC | 0.9909 |
| PR-AUC | 0.9979 |

Confusion matrix: 83,928 leaks caught, 1,194 missed, 2,956 false alarms, 17,042 correct no-leak. Top feature: `flow_mean_24h`.

Notes: about 81% of 2019 readings are inside a leak, and the 2019 test has only two leaks (one lasts 273 days), so accuracy alone is easy to reach. Separately, the average AUC-PR over 6 time-based folds of 2018 is 0.84 (weakest fold 0.34, best 0.99). Quiet mode gives 15.5 false-alert hours a month, Sensitive mode 43.3.

## IBM LinuxONE

The project ran on an IBM LinuxONE Community Cloud virtual server: Ubuntu 24.04, 3 vCPU, 8 GB RAM, architecture `s390x`.




Connect and run:

```bash
ssh -i key.pem linux1@<VM-IP>
uname -m          # prints s390x
```

To view the dashboard from your laptop, open a tunnel in a second terminal and browse to http://localhost:8501:

```bash
ssh -L 8501:localhost:8501 -i key.pem linux1@<VM-IP>
```

## Run it

Python 3.10 or newer.

```bash
git clone https://github.com/SaranyaVenkatasubramani/aqua.git
cd aqua
python3 -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python prepare_model.py
python -m streamlit run app.py
```

Open http://localhost:8501. The data file is expected at `data/original aqua.csv`; a compatible CSV can also be loaded with **Upload Data** on the Home page.

`prepare_model.py` creates `models/model.joblib`: it fits weekday/weekend profiles on leak-free 2018 rows, builds backward-only features, and fits a balanced `HistGradientBoostingClassifier` on 2018. If the model file is missing, the dashboard builds it on first launch.

## Dashboard pages

- **Home**: network KPIs and the data uploader.
- **Network**: schematic map of 33 pressure nodes, anomaly highlighting, observed vs expected pressure.
- **Detect**: live replay of the sensor stream, pressure and flow chart, leak-probability gauge, top suspect sensors.
- **Episodes**: leak episodes (runs merged across gaps under one hour), severity ranking, CSV export.
- **About**: pipeline, train/test split, 2019 metrics and confusion matrix.

<img width="825" height="657" alt="dashboard-detect" src="https://github.com/user-attachments/assets/f89f27e5-03e1-4439-b46a-46090a0fd38c" />

<img width="1518" height="725" alt="dashboard-network" src="https://github.com/user-attachments/assets/93a34d65-3308-4377-bb90-100f69bff58d" />


## Limits

- The 2019 test has only two leaks, and one lasts 273 days.
- Scores vary by period in 2018 (AUC-PR 0.34 to 0.99).
- The sensor pointer shows where pressure dropped most, not a confirmed leak location.
- Water saved is shown as zero until repair data is available.

## Data and open-source tools

- Data: BattLeDIM L-Town water network, 2018 to 2019 (https://github.com/BattLeDIM/BattLeDIM)
- Python, pandas, NumPy, scikit-learn, joblib, Streamlit, Matplotlib, seaborn
- Notebook: https://www.kaggle.com/code/dd1vya/aquasentinel

## Team Quantiq

Divya Lakshmi M, Divyadarshini M, Dharini S, Dhamini S, Saranya A V
