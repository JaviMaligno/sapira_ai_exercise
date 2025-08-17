from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import IsolationForest


ULB_PATH = Path("/home/javier/repos/datasets/ULB Credit Card Fraud Dataset (European Cardholders)/fraudTrain.csv")


@dataclass
class Config:
    row_limit: int = 200_000
    random_state: int = 42
    test_size: float = 0.2  # time-aware split will override
    out_dir: Path = Path("reports/phase2/ulb_gbdt")


def load_ulb(limit: int) -> pd.DataFrame:
    usecols = [
        "trans_num", "unix_time", "category", "amt", "merchant", "is_fraud", "cc_num"
    ]
    df = pd.read_csv(ULB_PATH, usecols=usecols, nrows=limit)
    df.rename(columns={
        "trans_num": "transaction_id",
        "unix_time": "event_time_ts",
        "category": "operation_type",
        "amt": "amount",
        "merchant": "merchant_name",
    }, inplace=True)
    df["event_time"] = pd.to_datetime(df["event_time_ts"], unit="s", utc=True)
    return df


def engineer(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = df.copy()
    df.sort_values("event_time", inplace=True)

    # Amounts
    df["abs_amount"] = df["amount"].abs()
    df["log1p_abs_amount"] = np.log1p(df["abs_amount"].clip(lower=0))

    # Time features
    df["hour"] = df["event_time"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow"] = df["event_time"].dt.dayofweek
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)

    # Category winsor z_iqr
    cap = df.groupby("operation_type")["amount"].transform(lambda s: s.quantile(0.995))
    df["amount_winsor_cat"] = np.minimum(df["amount"], cap)
    cat_median = df.groupby("operation_type")["amount_winsor_cat"].transform("median")
    cat_iqr = df.groupby("operation_type")["amount_winsor_cat"].transform(lambda s: (s.quantile(0.75) - s.quantile(0.25)))
    df["amount_z_iqr_cat"] = (df["amount_winsor_cat"] - cat_median) / cat_iqr.replace({0: np.nan})

    # Merchant sanitize
    s = df["merchant_name"].astype(str).str.lower()
    s = s.str.replace(r"^fraud_", "", regex=True)
    s = s.str.replace(r"[^a-z0-9\s]+", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    df["merchant_name"] = s

    # Simple velocities by card
    df["txn_count_24h"], df["time_since_last_sec"] = 0.0, 0.0
    df["is_new_merchant_for_card"] = 0
    for card, g in df.groupby("cc_num", sort=False):
        idx = g.index
        t = g["event_time"].values.astype('datetime64[ns]').astype('int64')
        win = 24 * 3600 * 1_000_000_000
        counts = np.zeros(len(g), dtype=float)
        j = 0
        for i in range(len(t)):
            while t[i] - t[j] > win and j < i:
                j += 1
            counts[i] = i - j + 1
        df.loc[idx, "txn_count_24h"] = counts
        # time since last
        last = None
        seen_merch = set()
        for ix, (tt, merch) in zip(idx, zip(g["event_time"].values, g["merchant_name"].fillna("UNK").values)):
            if last is not None:
                df.loc[ix, "time_since_last_sec"] = (pd.Timestamp(tt).to_datetime64() - pd.Timestamp(last).to_datetime64()).astype('timedelta64[s]').astype(float)
            else:
                df.loc[ix, "time_since_last_sec"] = np.nan
            last = tt
            # new merchant flag (per card)
            df.loc[ix, "is_new_merchant_for_card"] = 0 if merch in seen_merch else 1
            seen_merch.add(merch)

    num_cols = [
        "amount", "abs_amount", "log1p_abs_amount", "amount_z_iqr_cat",
        "txn_count_24h", "time_since_last_sec",
        "is_new_merchant_for_card",
        "hour", "hour_sin", "hour_cos", "dow", "is_night",
    ]
    cat_cols = ["operation_type", "merchant_name"]
    return df, num_cols, cat_cols


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k_frac: float) -> float:
    n = len(scores)
    k = max(1, int(n * k_frac))
    order = np.argsort(scores)[::-1]
    topk = order[:k]
    return float(np.sum(y_true[topk] == 1) / k)


def main():
    cfg = Config()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_ulb(cfg.row_limit)
    y = df["is_fraud"].astype(int).values
    df, num_cols, cat_cols = engineer(df)

    # Time-aware split (chronological)
    df_sorted = df.sort_values("event_time")
    split_idx = int(len(df_sorted) * 0.8)
    train_idx = df_sorted.index[:split_idx]
    test_idx = df_sorted.index[split_idx:]

    X_train = df.loc[train_idx, :]
    y_train = y[train_idx]
    X_test = df.loc[test_idx, :]
    y_test = y[test_idx]

    pre = ColumnTransformer(transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline(steps=[
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False)),
        ]), cat_cols),
    ])

    # Class weight by inverse prevalence
    pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    clf = HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_depth=8,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        l2_regularization=1e-3,
        random_state=cfg.random_state,
        class_weight={0: 1.0, 1: float(pos_weight)},
    )
    pipe = Pipeline(steps=[("prep", pre), ("gbdt", clf)])

    # Frequency encoding for merchant_name (fit on train only)
    freq_map = X_train["merchant_name"].fillna("UNK").value_counts(normalize=True).to_dict()
    X_train_ext = X_train.copy()
    X_test_ext = X_test.copy()
    X_train_ext["merchant_freq"] = X_train_ext["merchant_name"].fillna("UNK").map(freq_map).fillna(0.0)
    X_test_ext["merchant_freq"] = X_test_ext["merchant_name"].fillna("UNK").map(freq_map).fillna(0.0)

    # Isolation Forest as stacked feature (fit on train legitimate only)
    if_prep = ColumnTransformer(transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline(steps=[
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False)),
        ]), cat_cols),
    ])
    if_model = IsolationForest(n_estimators=200, max_samples="auto", contamination="auto", random_state=cfg.random_state, n_jobs=-1)
    if_pipe = Pipeline(steps=[("prep", if_prep), ("if", if_model)])
    X_train_legit = X_train[y_train == 0]
    if_pipe.fit(X_train_legit)
    # Compute anomaly scores for train/test
    train_if_score = -if_pipe.named_steps["if"].decision_function(if_pipe.named_steps["prep"].transform(X_train))
    test_if_score = -if_pipe.named_steps["if"].decision_function(if_pipe.named_steps["prep"].transform(X_test))
    # Simple rule score (train-derived thresholds)
    p995 = float(np.nanpercentile(X_train["abs_amount"], 99.5)) if X_train["abs_amount"].notna().any() else np.nan
    def compute_rule_score(dfp: pd.DataFrame) -> np.ndarray:
        high_amt = ((dfp["abs_amount"] >= p995) & dfp["abs_amount"].notna()).astype(int)
        new_merch_high = ((dfp["is_new_merchant_for_card"] == 1) & (dfp["abs_amount"] >= p995)).astype(int)
        rapid = (dfp["txn_count_24h"] >= 10).astype(int)
        return (1.5 * high_amt + 1.0 * new_merch_high + 0.5 * rapid).values
    train_rule = compute_rule_score(X_train)
    test_rule = compute_rule_score(X_test)

    # Append stacked features
    X_train_ext["if_anomaly_score"] = train_if_score
    X_test_ext["if_anomaly_score"] = test_if_score
    X_train_ext["rule_score"] = train_rule
    X_test_ext["rule_score"] = test_rule

    # Extend preprocess numeric columns to include stacked features
    num_cols_ext = num_cols + ["merchant_freq", "if_anomaly_score", "rule_score"]
    # Drop merchant_name from categoricals now that we added merchant_freq
    cat_cols_ext = [c for c in cat_cols if c != "merchant_name"]
    pre_ext = ColumnTransformer(transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols_ext),
        ("cat", Pipeline(steps=[
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False)),
        ]), cat_cols_ext),
    ])
    pipe_ext = Pipeline(steps=[("prep", pre_ext), ("gbdt", clf)])
    pipe_ext.fit(X_train_ext, y_train)

    # Scores and metrics
    if hasattr(pipe_ext.named_steps["gbdt"], "predict_proba"):
        scores = pipe_ext.named_steps["gbdt"].predict_proba(pipe_ext.named_steps["prep"].transform(X_test_ext))[:, 1]
    else:
        scores = pipe_ext.named_steps["gbdt"].predict(X_test_ext)
    ap = average_precision_score(y_test, scores)
    p_at_0_5 = precision_at_k(y_test, scores, 0.005)
    p_at_1_0 = precision_at_k(y_test, scores, 0.01)
    p_at_5_0 = precision_at_k(y_test, scores, 0.05)

    out_dir = cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "fraud_rate_train": float((y_train == 1).mean()),
        "fraud_rate_test": float((y_test == 1).mean()),
        "average_precision": float(ap),
        "precision_at": {
            "0.5%": float(p_at_0_5),
            "1.0%": float(p_at_1_0),
            "5.0%": float(p_at_5_0),
        },
    }
    (out_dir / "ulb_gbdt_summary.json").write_text(json.dumps(summary, indent=2))
    print("Wrote", out_dir / "ulb_gbdt_summary.json")


if __name__ == "__main__":
    main()


