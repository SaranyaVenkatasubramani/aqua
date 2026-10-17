from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "model.joblib"
LABEL_COL = "Leak_Status"
PUMP_COL = "PUMP_1"
FLOW_COLS = ["p227", "p235"]
N_SLOTS = 288
STD_FLOOR = 0.10


def find_dataset() -> Path:
    candidates = (
        DATA_DIR / "aquasentinal_dataset.csv",
        DATA_DIR / "aquasentinel_dataset.csv",
        DATA_DIR / "original aqua.csv",
    )
    for path in candidates:
        if path.is_file():
            return path
    expected = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "AquaSentinel dataset not found. Place the supplied CSV at one of:\n"
        f"{expected}"
    )


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Read the 5-minute sensor dataset and use its timestamp as the index."""
    csv_path = Path(path) if path else find_dataset()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {csv_path}")
    frame = pd.read_csv(csv_path, parse_dates=["Timestamp"])
    return validate_frame(frame, str(csv_path))


def load_dataset_bytes(content: bytes, name: str = "uploaded CSV") -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(content), parse_dates=["Timestamp"])
    return validate_frame(frame, name)


def validate_frame(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if "Timestamp" not in frame.columns:
        raise ValueError(f"{source_name} is missing the Timestamp column.")
    if frame.empty:
        raise ValueError(f"Dataset is empty: {source_name}")
    if frame["Timestamp"].isna().any():
        raise ValueError("The dataset contains invalid Timestamp values.")
    frame = frame.set_index("Timestamp").sort_index()
    if frame.index.has_duplicates:
        raise ValueError("The dataset contains duplicate timestamps.")
    required = [LABEL_COL, PUMP_COL, *FLOW_COLS]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    pressure_cols = [col for col in frame.columns if col.startswith("n")]
    if not pressure_cols:
        raise ValueError("No pressure sensor columns (n*) were found.")
    return frame


def fit_profiles(
    train_rows: pd.DataFrame,
    pressure_cols: list[str],
    flow_cols: list[str] = FLOW_COLS,
    label_col: str = LABEL_COL,
) -> dict[str, Any]:
    """Fit notebook-compatible weekday/weekend median profiles on clean rows."""
    clean = train_rows[train_rows[label_col] == 0].copy()
    if clean.empty:
        raise ValueError("No leak-free rows in training data; profiles cannot be fit.")

    clean["_slot"] = clean.index.hour * 12 + clean.index.minute // 5
    clean["_daytype"] = (clean.index.dayofweek >= 5).astype(int)
    clean["_flow"] = clean[flow_cols].sum(axis=1)

    profiles: dict[str, Any] = {
        "pressure": {},
        "flow": {},
        "std_floor": STD_FLOOR,
        "pressure_cols": list(pressure_cols),
    }

    for day_type in (0, 1):
        subset = clean[clean["_daytype"] == day_type]
        profiles["pressure"][day_type] = np.zeros((N_SLOTS, len(pressure_cols)))
        profiles["flow"][day_type] = np.zeros(N_SLOTS)
        for slot in range(N_SLOTS):
            slot_rows = subset[subset["_slot"] == slot]
            if not slot_rows.empty:
                profiles["pressure"][day_type][slot] = (
                    slot_rows[pressure_cols].median().values
                )
                profiles["flow"][day_type][slot] = slot_rows["_flow"].median()
            else:
                all_slot = clean[clean["_slot"] == slot]
                profiles["pressure"][day_type][slot] = (
                    all_slot[pressure_cols].median().values
                    if not all_slot.empty
                    else np.zeros(len(pressure_cols))
                )
                profiles["flow"][day_type][slot] = (
                    all_slot["_flow"].median() if not all_slot.empty else 0.0
                )
    return profiles


def build_features(
    df_in: pd.DataFrame,
    profiles: dict[str, Any],
    flow_cols: list[str] = FLOW_COLS,
    pump_col: str = PUMP_COL,
) -> pd.DataFrame:
    """Notebook feature builder: profile residuals and causal backward windows."""
    df_w = df_in.copy()
    pres_cols = profiles["pressure_cols"]
    std_floor = profiles["std_floor"]
    slots = (df_w.index.hour * 12 + df_w.index.minute // 5).astype(int)
    daytypes = (df_w.index.dayofweek >= 5).astype(int)

    profile_pressure = np.stack(
        [profiles["pressure"][0], profiles["pressure"][1]]
    )
    profile_flow = np.stack([profiles["flow"][0], profiles["flow"][1]])
    prof_pres = profile_pressure[daytypes, slots]
    prof_flow = profile_flow[daytypes, slots]

    pres_vals = df_w[pres_cols].values
    flow_total = df_w[flow_cols].sum(axis=1).values
    p227_vals = df_w["p227"].values
    p235_vals = df_w["p235"].values
    pump_vals = df_w[pump_col].values
    pres_resid = pres_vals - prof_pres
    flow_resid = flow_total - prof_flow

    feat = pd.DataFrame(index=df_w.index)
    feat["mean_pressure"] = pres_vals.mean(axis=1)
    feat["total_flow"] = flow_total
    feat["p227"] = p227_vals
    feat["p235"] = p235_vals
    feat["pump"] = pump_vals
    feat["pump_off"] = (pump_vals == 0).astype(float)

    tod_rad = (slots / N_SLOTS) * 2 * np.pi
    feat["sin_tod"] = np.sin(tod_rad)
    feat["cos_tod"] = np.cos(tod_rad)
    feat["resid_mean"] = pres_resid.mean(axis=1)
    feat["resid_min"] = pres_resid.min(axis=1)
    feat["resid_std"] = pres_resid.std(axis=1)
    feat["flow_resid"] = flow_resid

    for window_name, n_steps in (("1h", 12), ("6h", 72), ("24h", 288)):
        feat[f"resid_mean_roll_{window_name}"] = (
            feat["resid_mean"].rolling(n_steps, min_periods=1).mean()
        )
        feat[f"flow_resid_roll_{window_name}"] = (
            feat["flow_resid"].rolling(n_steps, min_periods=1).mean()
        )

    one_h_flow = feat["total_flow"].rolling(12, min_periods=1).mean()
    feat["night_flow_24h_min"] = one_h_flow.rolling(288, min_periods=1).min()
    one_h_pres = feat["mean_pressure"].rolling(12, min_periods=1).mean()
    feat["pres_24h_max"] = one_h_pres.rolling(288, min_periods=1).max()

    resid_df = pd.DataFrame(
        pres_resid,
        index=df_w.index,
        columns=[f"resid_{col}" for col in pres_cols],
    )
    result = pd.concat([feat, resid_df], axis=1)
    if std_floor != STD_FLOOR:
        raise ValueError(
            f"Unsupported profile std_floor={std_floor}; expected {STD_FLOOR}."
        )
    return result


def extract_episodes(
    series: pd.Series,
    gap_threshold: pd.Timedelta = pd.Timedelta("1h"),
) -> pd.DataFrame:
    """Merge adjacent positive-label runs separated by less than one hour."""
    values = series.copy().astype(int)
    changes = values.ne(values.shift())
    block_id = changes.cumsum()
    blocks = (
        values.groupby(block_id)
        .agg(
            value="first",
            start=lambda part: part.index[0],
            end=lambda part: part.index[-1],
        )
        .reset_index(drop=True)
    )
    leaks = blocks[blocks["value"] == 1].reset_index(drop=True)
    merged: list[dict[str, Any]] = []
    i = 0
    while i < len(leaks):
        start = leaks.loc[i, "start"]
        end = leaks.loc[i, "end"]
        while i + 1 < len(leaks):
            gap = leaks.loc[i + 1, "start"] - end
            if gap < gap_threshold:
                end = leaks.loc[i + 1, "end"]
                i += 1
            else:
                break
        merged.append(
            {
                "start": start,
                "end": end,
                "days": (end - start).total_seconds() / 86400,
            }
        )
        i += 1
    return pd.DataFrame(merged, columns=["start", "end", "days"])


def train_bundle(
    frame: pd.DataFrame,
    output_path: str | Path = MODEL_PATH,
) -> dict[str, Any]:
    """Fit the notebook's final 2018 model and serialize its inference bundle."""
    pressure_cols = [col for col in frame.columns if col.startswith("n")]
    train_mask = frame.index.year == 2018
    if not train_mask.any() or not (frame.index.year == 2019).any():
        raise ValueError("Training requires the notebook's 2018/2019 temporal split.")

    profiles = fit_profiles(frame.loc[train_mask], pressure_cols)
    features = build_features(frame, profiles)
    feature_cols = list(features.columns)
    model = HistGradientBoostingClassifier(
        max_iter=150,
        max_depth=6,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(
        features.loc[train_mask, feature_cols].values.astype(np.float32),
        frame.loc[train_mask, LABEL_COL].astype(int).values,
    )
    clean_train = train_mask & (frame[LABEL_COL].values == 0)
    residual_cols = [f"resid_{sensor}" for sensor in pressure_cols]
    residual_std = (
        features.loc[clean_train, residual_cols].std().clip(lower=profiles["std_floor"])
    )
    bundle = {
        "model": model,
        "profiles": profiles,
        "feature_cols": feature_cols,
        "residual_std": residual_std,
        "thresholds": {"sensitive": 0.5, "quiet": 0.9},
        "sklearn_version": sklearn.__version__,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return bundle


def load_or_train_bundle(
    frame: pd.DataFrame,
    model_path: str | Path = MODEL_PATH,
) -> dict[str, Any]:
    """Load the notebook bundle or build it when the artifact was not supplied."""
    path = Path(model_path)
    if path.is_file():
        bundle = joblib.load(path)
        required = {"model", "profiles", "feature_cols", "thresholds"}
        missing = required.difference(bundle)
        if missing:
            raise ValueError(
                f"Model bundle {path} is missing keys: {', '.join(sorted(missing))}"
            )
        if bundle.get("sklearn_version") != sklearn.__version__:
            raise RuntimeError(
                "The model was serialized with scikit-learn "
                f"{bundle.get('sklearn_version')}, but {sklearn.__version__} is "
                "installed. Rebuild it with `python prepare_model.py`."
            )
        return bundle
    return train_bundle(frame, path)
