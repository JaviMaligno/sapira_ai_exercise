from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ULB_PATH = \
    Path("/home/javier/repos/datasets/ULB Credit Card Fraud Dataset (European Cardholders)/fraudTrain.csv")


@dataclass
class EvalConfig:
    row_limit: int = 200_000
    random_state: int = 42
    out_dir: Path = Path("reports/phase1/ulb_eval")


def load_ulb(row_limit: int) -> pd.DataFrame:
    usecols = [
        "trans_num", "unix_time", "category", "amt", "merchant", "is_fraud", "cc_num"
    ]
    df = pd.read_csv(ULB_PATH, usecols=usecols, nrows=row_limit)
    df.rename(columns={
        "trans_num": "transaction_id",
        "unix_time": "event_time_ts",
        "category": "operation_type",
        "amt": "amount",
        "merchant": "merchant_name",
    }, inplace=True)
    df["event_time"] = pd.to_datetime(df["event_time_ts"], unit="s", utc=True)
    return df


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = df.copy()
    df.sort_values("event_time", inplace=True)

    # Basic numeric features
    df["abs_amount"] = df["amount"].abs()
    df["log1p_abs_amount"] = np.log1p(df["abs_amount"].clip(lower=0))
    df["hour"] = df["event_time"].dt.hour
    df["dow"] = df["event_time"].dt.dayofweek
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)

    # Rolling counts by card (cc_num)
    def rolling_count_by_hours(times: pd.Series, window_hours: int) -> np.ndarray:
        n = len(times)
        counts = np.zeros(n, dtype=float)
        mask = times.notna().values
        if not mask.any():
            return counts
        idxs = np.where(mask)[0]
        t = times.values[mask].astype('datetime64[ns]').astype('int64')
        win = np.int64(window_hours) * np.int64(3600_000_000_000)  # ns
        j = 0
        for i in range(len(t)):
            while t[i] - t[j] > win and j < i:
                j += 1
            counts[idxs[i]] = (i - j + 1)
        return counts

    if "cc_num" in df.columns:
        df.sort_values(["cc_num", "event_time"], inplace=True)
        df["txn_count_1h"] = 0.0
        df["txn_count_24h"] = 0.0
        for card, g in df.groupby("cc_num", sort=False, dropna=False):
            idx = g.index
            c1 = rolling_count_by_hours(g["event_time"], 1)
            c24 = rolling_count_by_hours(g["event_time"], 24)
            df.loc[idx, "txn_count_1h"] = c1
            df.loc[idx, "txn_count_24h"] = c24
        # Time since last per card
        df["time_since_last_sec"] = 0.0
        last_ts = {}
        vals = np.zeros(len(df), dtype=float)
        for i, (card, t) in enumerate(zip(df["cc_num"].values, df["event_time"].values)):
            prev = last_ts.get(card)
            if pd.notna(t) and prev is not None:
                vals[i] = (pd.Timestamp(t).to_datetime64() - pd.Timestamp(prev).to_datetime64()).astype('timedelta64[s]').astype(float)
            else:
                vals[i] = np.nan
            last_ts[card] = pd.Timestamp(t) if pd.notna(t) else last_ts.get(card)
        df["time_since_last_sec"] = vals
    else:
        df["txn_count_1h"] = 0.0
        df["txn_count_24h"] = 0.0
        df["time_since_last_sec"] = 0.0

    numeric_features = [
        "amount", "abs_amount", "log1p_abs_amount", "hour", "dow", "is_night",
        "txn_count_1h", "txn_count_24h", "time_since_last_sec",
    ]
    categorical_features = ["operation_type", "merchant_name"]
    return df, numeric_features, categorical_features


def build_if_pipeline(numeric_features: List[str], categorical_features: List[str], random_state: int) -> Pipeline:
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])
    model = IsolationForest(
        n_estimators=400, max_samples="auto", contamination="auto", random_state=random_state, n_jobs=-1
    )
    return Pipeline(steps=[("prep", preprocessor), ("model", model)])


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k_frac: float) -> float:
    n = len(scores)
    k = max(1, int(n * k_frac))
    order = np.argsort(scores)[::-1]
    topk = order[:k]
    return float(np.sum(y_true[topk] == 1) / k)


def main():
    cfg = EvalConfig()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ULB data...")
    df = load_ulb(cfg.row_limit)
    y = df["is_fraud"].astype(int).values
    print(f"Loaded {len(df):,} rows (fraud rate ~ {y.mean():.4%}). Engineering features...")

    df, num_cols, cat_cols = engineer_features(df)

    print("Training Isolation Forest on legitimate samples...")
    mask_legit = (y == 0)
    pipe = build_if_pipeline(num_cols, cat_cols, cfg.random_state)
    # Contamination guard: exclude top 0.5% by abs amount when fitting IF
    subset = df.loc[mask_legit, :].copy()
    p995 = float(np.percentile(subset["abs_amount"].dropna(), 99.5)) if subset["abs_amount"].notna().any() else np.inf
    subset = subset[subset["abs_amount"] < p995]
    pipe.fit(subset)

    print("Scoring anomaly scores...")
    decision = pipe.decision_function(df)  # higher = more normal
    anomaly_score = -decision

    print("Evaluating metrics...")
    # Higher anomaly_score => more likely fraud, suitable for PR evaluation
    ap = average_precision_score(y, anomaly_score)
    prec, rec, _ = precision_recall_curve(y, anomaly_score)

    p_at_0_1 = precision_at_k(y, anomaly_score, 0.001)
    p_at_0_5 = precision_at_k(y, anomaly_score, 0.005)
    p_at_1_0 = precision_at_k(y, anomaly_score, 0.01)

    scores_out = cfg.out_dir / "ulb_if_scores.csv"
    pd.DataFrame({
        "transaction_id": df["transaction_id"],
        "is_fraud": y,
        "anomaly_score": anomaly_score,
    }).to_csv(scores_out, index=False)

    summary = {
        "rows": int(len(df)),
        "fraud_rate": float(y.mean()),
        "average_precision": float(ap),
        "precision_at": {
            "0.1%": p_at_0_1,
            "0.5%": p_at_0_5,
            "1.0%": p_at_1_0,
        },
    }
    (cfg.out_dir / "ulb_if_eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {scores_out} and summary json with AP={ap:.4f}, P@0.5%={p_at_0_5:.4f}")


if __name__ == "__main__":
    main()


