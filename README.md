# AquaSentinel

A Streamlit dashboard for visualizing the supplied municipal water-network
readings, replaying causal leak scores, inspecting pressure nodes, and exploring
label-derived leak episodes.

## Run locally

Use Python 3.10 or newer. From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python prepare_model.py
python -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`).
The supplied CSV is expected at `data\original aqua.csv`; the app also accepts
`data\aquasentinal_dataset.csv` or `data\aquasentinel_dataset.csv`. A compatible
CSV can also be loaded from the Home page's **Upload Data** control.

`prepare_model.py` creates `models\model.joblib`. The uploaded materials
contained the 2018–2019 CSV and the source notebook, but not the notebook's
serialized joblib bundle. The script therefore reproduces the notebook's final
training recipe: fit weekday/weekend profiles on leak-free 2018 rows, construct
the same backward-only feature set, fit the balanced
`HistGradientBoostingClassifier` on 2018, and save its profiles/model/catalogue
and clean-training residual scales in the inference bundle. If the model file
is absent, the dashboard builds it on first launch; running the script first
makes that work explicit.

## Dashboard pages

- **Home** — network KPIs and entry points to the monitor and data uploader.
- **Network** — schematic pressure-node map (33 channels in the supplied CSV),
  anomaly highlighting, and selected-node
  pressure compared with its expected weekday/weekend daily profile.
- **Detect** — historical stream playback, pressure/flow chart with predicted
  alert intervals, leak-probability gauge, and peer-normalized suspect ranking.
- **Episodes** — `Leak_Status` runs merged across gaps shorter than one hour,
  timeline, severity ranking, and CSV export.
- **About** — pipeline, temporal-split explanation, 2019 holdout metrics, and
  confusion matrix.

The model's classifier features follow the notebook. Peer-normalized
sensor-z-scores (with a 0.10 standard-deviation floor) are used for suspect
ranking, not substituted for the notebook's profile-residual feature set.
Water-saved volume is displayed as zero until repair/intervention telemetry is
available; it is not inferred from sensor data alone.

The dashboard caches its loaded dataset, trained model, engineered features,
and scores in Streamlit's resource cache. Clear the app cache after replacing
the CSV or model bundle.
