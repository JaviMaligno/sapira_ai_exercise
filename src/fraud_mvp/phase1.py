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
import json as _json
from datetime import datetime, timezone

@dataclass
class Phase1Config:
    limit_docs: int = 100_000
    anomaly_rate_target: float = 0.005  # 0.5% alerts by anomaly threshold
    random_state: int = 42
    out_dir: Path = Path("reports/phase1")
    per_category_thresholds_path: Path = Path("reports/phase1/ulb_eval/ulb_if_per_category_thresholds.json")
    merchant_vocab_path: Path = Path("reports/phase1/merchant_vocab.json")


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
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
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
    # New category flag and time since last in category per account
    df["is_new_category_for_account"] = 0
    df["time_since_last_cat_sec"] = 0.0
    # Category transition rarity per account (lower frequency -> higher rarity)
    df["cat_transition_rarity"] = 0.0
    # Merchant frequency ratio/rarity within account
    df["merchant_freq_ratio_account"] = 0.0
    df["merchant_rare_score_account"] = 0.0
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
            # Category novelty/time tracking
            seen_cat: Dict[str, bool] = {}
            last_cat_ts: Dict[str, np.int64] = {}
            # Transition counts (prev_cat -> cur_cat)
            trans_counts: Dict[Tuple[str, str], int] = {}
            prev_cat: Optional[str] = None
            # Merchant cumulative counts within account
            merch_counts: Dict[str, int] = {}
            total_seen = 0
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
                # Per-account category novelty/time since last and transition rarity
                cat = df.at[idx, "operation_type"] if "operation_type" in df.columns else "UNK"
                df.loc[idx, "is_new_category_for_account"] = 0 if cat in seen_cat else 1
                if cat in last_cat_ts and tsec is not None:
                    df.loc[idx, "time_since_last_cat_sec"] = float(tsec - last_cat_ts[cat])
                else:
                    df.loc[idx, "time_since_last_cat_sec"] = np.nan
                # Transition rarity
                if prev_cat is None:
                    df.loc[idx, "cat_transition_rarity"] = 0.0
                else:
                    key = (prev_cat, cat)
                    count = trans_counts.get(key, 0)
                    df.loc[idx, "cat_transition_rarity"] = 1.0 / (count + 1.0)
                    trans_counts[key] = count + 1
                seen_cat[cat] = True
                if tsec is not None:
                    last_cat_ts[cat] = tsec
                prev_cat = cat
                # Merchant frequency within account
                total_seen += 1
                mc = merch_counts.get(merch, 0)
                df.loc[idx, "merchant_freq_ratio_account"] = float(mc) / float(total_seen)
                df.loc[idx, "merchant_rare_score_account"] = 1.0 / float(mc + 1)
                merch_counts[merch] = mc + 1
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

    # Category-aware robust z-score (IQR-based) when operation_type present
    if "operation_type" in df.columns:
        # Winsorize per category at p99.5 and compute robust z
        cap = df.groupby("operation_type")["amount"].transform(lambda s: s.quantile(0.995))
        df["amount_winsor_cat"] = np.minimum(df["amount"], cap)
        cat_median = df.groupby("operation_type")["amount_winsor_cat"].transform("median")
        def _iqr(s: pd.Series) -> float:
            return s.quantile(0.75) - s.quantile(0.25)
        cat_iqr = df.groupby("operation_type")["amount_winsor_cat"].transform(_iqr)
        df["amount_z_iqr_cat"] = (df["amount_winsor_cat"] - cat_median) / cat_iqr.replace({0: np.nan})
    else:
        df["amount_z_iqr_cat"] = np.nan

    numeric_features = [
        "amount",
        "abs_amount",
        "log1p_abs_amount",
        "hour",
        "hour_sin",
        "hour_cos",
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
        "amount_z_iqr_cat",
        "merchant_txn_count_7d",
        "merchant_txn_count_30d",
        "is_new_category_for_account",
        "time_since_last_cat_sec",
        "cat_transition_rarity",
    ]
    categorical_features = [
        "operation_type",
        "merchant_name",
        "currency",
    ]

    return df, numeric_features, categorical_features


def build_if_pipeline(numeric_features: List[str], categorical_features: List[str], random_state: int,
                      category_vocab: Optional[Dict[str, List[str]]] = None,
                      n_estimators: int = 200,
                      max_features: float | int = 1.0) -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    # Determine categories spec for OneHotEncoder
    categories: "auto" | List[List[str]]
    if category_vocab is not None:
        ops = category_vocab.get("operation_type")
        merch = category_vocab.get("merchant_name")
        curr = category_vocab.get("currency")
        if isinstance(ops, list) and len(ops) > 0 and isinstance(merch, list) and len(merch) > 0 and isinstance(curr, list) and len(curr) > 0:
            categories = [ops, merch, curr]
        else:
            categories = "auto"
    else:
        categories = "auto"

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", categories=categories, min_frequency=None)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples="auto",
        max_features=max_features,
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
    # Apply ablation-driven recommended feature stack: amount_core + velocity + cat_robust + novelty_cat
    recommended_num = [
        "amount", "abs_amount", "log1p_abs_amount",  # amount_core
        "txn_count_1h", "txn_count_24h", "time_since_last_sec",  # velocity
        "amount_z_iqr_cat",  # cat_robust
        "is_new_category_for_account", "time_since_last_cat_sec",  # novelty_cat
    ]
    # Use intersection with available columns to avoid errors
    num_cols = [c for c in recommended_num if c in df.columns]

    print("Precomputing rule scores for contamination guard...")
    df = compute_rule_scores(df)

    print("Preparing categorical vocabularies...")
    cat_vocab: Dict[str, List[str]] = {}
    if "operation_type" in df.columns:
        cat_vocab["operation_type"] = sorted(df["operation_type"].dropna().unique().tolist())
    if "currency" in df.columns:
        cat_vocab["currency"] = sorted(df["currency"].dropna().unique().tolist())
    # Merchant vocab: load if exists, else build top-N frequent
    merchant_vocab: Optional[List[str]] = None
    try:
        if cfg.merchant_vocab_path.exists():
            merchant_vocab = _json.loads(cfg.merchant_vocab_path.read_text())
        else:
            freq = df["merchant_name"].fillna("UNK").value_counts()
            merchant_vocab = freq[freq >= 10].index.tolist()
            cfg.merchant_vocab_path.parent.mkdir(parents=True, exist_ok=True)
            cfg.merchant_vocab_path.write_text(_json.dumps(merchant_vocab, indent=2))
    except Exception:
        merchant_vocab = None
    if merchant_vocab is not None:
        cat_vocab["merchant_name"] = merchant_vocab

    print("Training Isolation Forest pipeline (excluding extreme rule hits)...")
    pipe = build_if_pipeline(num_cols, cat_cols, cfg.random_state, category_vocab=cat_vocab,
                             n_estimators=200, max_features=1.0)
    train_mask = ~((df["rule_extreme_amount"] == 1) | (df["rule_high_amount"] == 1))
    pipe.fit(df.loc[train_mask, :])

    print("Scoring anomaly scores...")
    decision = pipe.decision_function(df)  # higher = more normal
    anomaly_score = -decision
    df["anomaly_score"] = anomaly_score

    # Threshold by target alert rate (global and per-category, using ULB thresholds if present)
    tau = float(np.quantile(anomaly_score, 1 - cfg.anomaly_rate_target))
    # Sanity guard: avoid negative/unstable tau by clamping to a positive global quantile if needed
    if tau <= 0:
        pos_scores = anomaly_score[anomaly_score > 0]
        if pos_scores.size > 0:
            tau = float(np.quantile(pos_scores, 1 - cfg.anomaly_rate_target))
        else:
            tau = float(np.quantile(anomaly_score, 1 - min(cfg.anomaly_rate_target * 0.5, 0.001)))
    per_cat_tau: Dict[str, float] = {}
    # Load precomputed thresholds from ULB eval if available
    try:
        if cfg.per_category_thresholds_path.exists():
            import json as _json
            per_cat_tau = _json.loads(cfg.per_category_thresholds_path.read_text())
    except Exception:
        per_cat_tau = {}
    if "operation_type" in df.columns:
        # Fallback: compute from DGuard slice for categories not found in ULB file
        for cat, g in df.groupby("operation_type"):
            c = str(cat)
            if c not in per_cat_tau and len(g) >= 50:
                local_tau = float(np.quantile(g["anomaly_score"], 1 - cfg.anomaly_rate_target))
                # Guard: clamp negative thresholds to global tau
                if local_tau <= 0:
                    local_tau = tau
                per_cat_tau[c] = local_tau
        df["_tau_cat"] = df["operation_type"].map(per_cat_tau).fillna(tau)
        df["anomaly_alert"] = (df["anomaly_score"] >= df["_tau_cat"]).astype(int)
    else:
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
    # Cap per (account, merchant, day) and then per (account, day)
    if "account_id" in alerts.columns and "event_time" in alerts.columns:
        alerts["event_date"] = alerts["event_time"].dt.date
        if "merchant_name" in alerts.columns:
            alerts = (
                alerts.groupby(["account_id", "event_date", "merchant_name"], as_index=False, group_keys=False)
                .head(2)
            )
        alerts = (
            alerts.sort_values(["account_id", "event_date", "anomaly_score", "rule_score"], ascending=[True, True, False, False])
            .groupby(["account_id", "event_date"], as_index=False, group_keys=False)
            .head(3)
        )

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

    # Alerts per category summary
    cat_counts: Dict[str, int] = {}
    if "operation_type" in alerts.columns:
        cat_counts = alerts["operation_type"].fillna("UNK").value_counts().to_dict()

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
        "alerts_per_category": cat_counts,
    }
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Wrote alerts to {out_csv} and summary to {out_json}")

    # Drift metrics logging
    try:
        drift_dir = cfg.out_dir / "drift"
        drift_dir.mkdir(parents=True, exist_ok=True)
        report_date = datetime.now(timezone.utc).strftime('%Y%m%d')
        drift = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_rows": int(len(df)),
            "new_merchant_share": float((df.get("is_new_merchant_for_account", 0) == 1).mean()),
            "missing_rates": {
                "amount": float(df["amount"].isna().mean()),
                "operation_type": float(df.get("operation_type").isna().mean() if "operation_type" in df.columns else 0.0),
                "merchant_name": float(df.get("merchant_name").isna().mean() if "merchant_name" in df.columns else 0.0),
            },
            "amount_quantiles_by_category": {},
        }
        if "operation_type" in df.columns:
            for cat, g in df.groupby("operation_type"):
                drift["amount_quantiles_by_category"][str(cat)] = {
                    "p50": float(g["amount"].quantile(0.5)),
                    "p90": float(g["amount"].quantile(0.9)),
                    "p99": float(g["amount"].quantile(0.99)),
                }
        # Write dated file and append to log
        dated_path = drift_dir / f"drift_{report_date}.json"
        dated_path.write_text(json.dumps(drift, indent=2))
        with (drift_dir / "drift_log.jsonl").open('a') as flog:
            flog.write(json.dumps(drift) + "\n")
        print(f"Wrote drift metrics to {dated_path}")
    except Exception as e:
        print("Warning: drift logging failed:", e)


if __name__ == "__main__":
    main()


