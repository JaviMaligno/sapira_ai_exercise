from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pymongo import MongoClient
from sapira_etl.settings import MONGO_URI
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from collections import deque

@dataclass
class Phase1Config:
    limit_docs: int = 100_000
    anomaly_rate_target: float = 0.005  # 0.5% alerts by anomaly threshold
    random_state: int = 42
    out_dir: Path = Path("reports/phase1")


def fetch_dguard_transactions(limit_docs: int) -> pd.DataFrame:
    client = MongoClient(MONGO_URI, authSource="admin", serverSelectionTimeoutMS=10000)
    dbname = (MONGO_URI.rsplit('/', 1)[-1] or 'dguard_transactions').split('?')[0]
    coll = client[dbname]["bank_transactions"]
    fields = {
        "uuid": 1,
        "user_id": 1,
        "account_id": 1,
        "operation_date": 1,
        "amount": 1,
        "balance": 1,
        "currency": 1,
        "description": 1,
        "operation_type": 1,
        "merchant_clean_name": 1,
        "categories": 1,
    }
    docs = list(coll.find({}, fields).limit(limit_docs))
    client.close()
    df = pd.DataFrame(docs)
    if not df.empty:
        df.rename(
            columns={
                "uuid": "transaction_id",
                "operation_date": "event_time",
                "merchant_clean_name": "merchant_name",
            },
            inplace=True,
        )
        df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce", utc=True)
    return df


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = df.copy()
    df.sort_values("event_time", inplace=True)

    # Basic numeric features
    df["abs_amount"] = df["amount"].abs()
    df["log1p_abs_amount"] = np.log1p(df["abs_amount"].clip(lower=0))

    # Time features
    df["hour"] = df["event_time"].dt.hour
    df["dow"] = df["event_time"].dt.dayofweek
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)

    # Time since last transaction per account (seconds)
    df["time_since_last_sec"] = 0.0
    if "account_id" in df.columns:
        df.sort_values(["account_id", "event_time"], inplace=True)
        last_time = None
        last_by_acct: Dict[str, Optional[pd.Timestamp]] = {}
        vals = np.zeros(len(df), dtype=float)
        for i, (acct, t) in enumerate(zip(df["account_id"].values, df["event_time"].values)):
            prev = last_by_acct.get(acct)
            if pd.notna(t) and prev is not None and pd.notna(prev):
                vals[i] = (pd.Timestamp(t).to_datetime64() - pd.Timestamp(prev).to_datetime64()).astype('timedelta64[s]').astype(float)
            else:
                vals[i] = np.nan
            last_by_acct[acct] = pd.Timestamp(t) if pd.notna(t) else last_by_acct.get(acct)
        df["time_since_last_sec"] = vals

    # Simple rolling counts per account_id using two-pointer (handles NaT)
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

    if "account_id" in df.columns:
        df.sort_values(["account_id", "event_time"], inplace=True)
        df["txn_count_1h"] = 0.0
        df["txn_count_24h"] = 0.0
        for acct, g in df.groupby("account_id", sort=False, dropna=False):
            idx = g.index
            c1 = rolling_count_by_hours(g["event_time"], 1)
            c24 = rolling_count_by_hours(g["event_time"], 24)
            df.loc[idx, "txn_count_1h"] = c1
            df.loc[idx, "txn_count_24h"] = c24
    else:
        df["txn_count_1h"] = 0.0
        df["txn_count_24h"] = 0.0

    # New merchant flag and seen-within-7d per account
    df["is_new_merchant_for_account"] = 0
    df["merchant_seen_7d"] = 0
    df["merchant_txn_count_7d"] = 0.0
    df["merchant_txn_count_30d"] = 0.0
    if "account_id" in df.columns:
        df.sort_values(["account_id", "event_time"], inplace=True)
        seven_days = np.int64(7 * 24 * 3600)
        thirty_days = np.int64(30 * 24 * 3600)
        flags_new = np.zeros(len(df), dtype=int)
        flags_7d = np.zeros(len(df), dtype=int)
        c7 = np.zeros(len(df), dtype=float)
        c30 = np.zeros(len(df), dtype=float)
        for acct, g in df.groupby("account_id", sort=False, dropna=False):
            seen: Dict[str, int] = {}
            last_ts: Dict[str, np.int64] = {}
            win7: Dict[str, deque] = {}
            win30: Dict[str, deque] = {}
            for i, (idx, merch, t) in enumerate(zip(g.index, g["merchant_name"].fillna("UNK").values, g["event_time"].values)):
                tsec = pd.Timestamp(t).value // 10**9 if pd.notna(t) else None
                is_new = 0 if merch in seen else 1
                within7d = 0
                if merch in last_ts and tsec is not None:
                    within7d = 1 if (tsec - last_ts[merch]) <= seven_days else 0
                local_i = list(g.index).index(idx)
                flags_new[local_i] = is_new
                flags_7d[local_i] = within7d
                if tsec is not None:
                    dq7 = win7.setdefault(merch, deque())
                    dq30 = win30.setdefault(merch, deque())
                    while dq7 and (tsec - dq7[0]) > seven_days:
                        dq7.popleft()
                    while dq30 and (tsec - dq30[0]) > thirty_days:
                        dq30.popleft()
                    c7[local_i] = float(len(dq7))
                    c30[local_i] = float(len(dq30))
                    dq7.append(tsec)
                    dq30.append(tsec)
                seen[merch] = seen.get(merch, 0) + 1
                if tsec is not None:
                    last_ts[merch] = tsec
            # Assign flags
            df.loc[g.index, "is_new_merchant_for_account"] = flags_new[: len(g.index)]
            df.loc[g.index, "merchant_seen_7d"] = flags_7d[: len(g.index)]
            df.loc[g.index, "merchant_txn_count_7d"] = c7[: len(g.index)]
            df.loc[g.index, "merchant_txn_count_30d"] = c30[: len(g.index)]

    # Per-account aggregate amount stats (overall)
    if "account_id" in df.columns:
        agg = df.groupby("account_id")["abs_amount"].agg(["mean", "std", "median"])
        agg["p90"] = df.groupby("account_id")["abs_amount"].quantile(0.9)
        agg["p99"] = df.groupby("account_id")["abs_amount"].quantile(0.99)
        df = df.merge(agg, left_on="account_id", right_index=True, how="left", suffixes=(None, None))
        df.rename(columns={"mean": "acct_amt_mean", "std": "acct_amt_std", "median": "acct_amt_p50"}, inplace=True)
        df["amount_zscore"] = (df["abs_amount"] - df["acct_amt_mean"]) / df["acct_amt_std"].replace({0: np.nan})
    else:
        df["acct_amt_mean"] = np.nan
        df["acct_amt_std"] = np.nan
        df["acct_amt_p50"] = np.nan
        df["p90"] = np.nan
        df["p99"] = np.nan
        df["amount_zscore"] = np.nan

    numeric_features = [
        "amount",
        "abs_amount",
        "log1p_abs_amount",
        "hour",
        "dow",
        "is_night",
        "txn_count_1h",
        "txn_count_24h",
        "time_since_last_sec",
        "is_new_merchant_for_account",
        "merchant_seen_7d",
        "acct_amt_mean",
        "acct_amt_std",
        "acct_amt_p50",
        "p90",
        "p99",
        "amount_zscore",
        "merchant_txn_count_7d",
        "merchant_txn_count_30d",
    ]
    categorical_features = [
        "operation_type",
        "merchant_name",
        "currency",
    ]

    return df, numeric_features, categorical_features


def build_if_pipeline(numeric_features: List[str], categorical_features: List[str], random_state: int) -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = IsolationForest(
        n_estimators=400,
        max_samples="auto",
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    return pipe


def compute_rule_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Global percentiles for amount (robustness: compute on sample if needed)
    amt = df["abs_amount"].dropna()
    p995 = float(np.percentile(amt, 99.5)) if not amt.empty else np.nan
    p999 = float(np.percentile(amt, 99.9)) if not amt.empty else np.nan

    df["rule_high_amount"] = ((df["abs_amount"] >= p995) & df["abs_amount"].notna()).astype(int)
    df["rule_extreme_amount"] = ((df["abs_amount"] >= p999) & df["abs_amount"].notna()).astype(int)
    df["rule_new_merchant_high_amount"] = (
        (df["is_new_merchant_for_account"] == 1) & (df["abs_amount"] >= p995)
    ).astype(int)

    # Velocity-based simple rules (approximate)
    df["rule_rapid_repeats"] = (df["txn_count_1h"] >= 5).astype(int)

    # Z-score extreme
    df["rule_amount_z3"] = ((df["amount_zscore"] >= 3.0) & df["amount_zscore"].notna()).astype(int)

    # Aggregate rule score
    df["rule_score"] = (
        2.0 * df["rule_extreme_amount"]
        + 1.5 * df["rule_high_amount"]
        + 1.0 * df["rule_new_merchant_high_amount"]
        + 0.5 * df["rule_rapid_repeats"]
        + 1.5 * df["rule_amount_z3"]
    )
    return df


def main():
    cfg = Phase1Config()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching DGuard transactions...")
    df = fetch_dguard_transactions(cfg.limit_docs)
    if df.empty:
        print("No data fetched from Mongo.")
        return

    print(f"Fetched {len(df):,} rows. Engineering features...")
    df, num_cols, cat_cols = engineer_features(df)

    print("Precomputing rule scores for contamination guard...")
    df = compute_rule_scores(df)

    print("Training Isolation Forest pipeline (excluding extreme rule hits)...")
    pipe = build_if_pipeline(num_cols, cat_cols, cfg.random_state)
    train_mask = ~((df["rule_extreme_amount"] == 1) | (df["rule_high_amount"] == 1))
    pipe.fit(df.loc[train_mask, :])

    print("Scoring anomaly scores...")
    decision = pipe.decision_function(df)  # higher = more normal
    anomaly_score = -decision
    df["anomaly_score"] = anomaly_score

    # Threshold by target alert rate
    tau = float(np.quantile(anomaly_score, 1 - cfg.anomaly_rate_target))
    df["anomaly_alert"] = (df["anomaly_score"] >= tau).astype(int)

    # rule scores already computed

    print("Combining decisions...")
    # Basic hybrid policy
    df["final_alert"] = (
        (df["rule_score"] >= 2) | (df["anomaly_alert"] == 1) | ((df["rule_score"] > 0) & (df["anomaly_score"] >= tau * 0.9))
    ).astype(int)

    # Persist outputs
    alerts = df[df["final_alert"] == 1].copy()
    alerts = alerts.sort_values(["anomaly_score", "rule_score"], ascending=[False, False])

    out_csv = cfg.out_dir / "alerts_phase1.csv"
    out_json = cfg.out_dir / "alerts_phase1_summary.json"
    alerts_cols = [
        "transaction_id",
        "event_time",
        "user_id",
        "account_id",
        "amount",
        "operation_type",
        "merchant_name",
        "anomaly_score",
        "rule_score",
        "final_alert",
    ]
    alerts[alerts_cols].to_csv(out_csv, index=False)

    summary = {
        "total_rows": int(len(df)),
        "alerts": int(len(alerts)),
        "alert_rate": float(len(alerts) / len(df)),
        "tau": tau,
        "top_rules": {
            "extreme_amount": int(df["rule_extreme_amount"].sum()),
            "high_amount": int(df["rule_high_amount"].sum()),
            "new_merchant_high": int(df["rule_new_merchant_high_amount"].sum()),
            "rapid_repeats": int(df["rule_rapid_repeats"].sum()),
        },
    }
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Wrote alerts to {out_csv} and summary to {out_json}")


if __name__ == "__main__":
    main()


