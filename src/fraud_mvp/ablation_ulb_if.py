from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ULB_PATH = Path("/home/javier/repos/datasets/ULB Credit Card Fraud Dataset (European Cardholders)/fraudTrain.csv")


@dataclass
class AblationConfig:
    row_limit: int = 200_000
    random_state: int = 42
    out_dir: Path = Path("reports/phase1/ulb_eval")
    n_estimators: int = 200
    max_features: float | int = 1.0


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


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]], List[str]]:
    df = df.copy()
    df.sort_values("event_time", inplace=True)

    # Core
    df["abs_amount"] = df["amount"].abs()
    df["log1p_abs_amount"] = np.log1p(df["abs_amount"].clip(lower=0))

    # Time
    df["hour"] = df["event_time"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow"] = df["event_time"].dt.dayofweek
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)

    # Rolling helper
    def rolling_count_by_hours(times: pd.Series, window_hours: int) -> np.ndarray:
        n = len(times)
        counts = np.zeros(n, dtype=float)
        mask = times.notna().values
        if not mask.any():
            return counts
        idxs = np.where(mask)[0]
        t = times.values[mask].astype('datetime64[ns]').astype('int64')
        win = np.int64(window_hours) * np.int64(3600_000_000_000)
        j = 0
        for i in range(len(t)):
            while t[i] - t[j] > win and j < i:
                j += 1
            counts[idxs[i]] = (i - j + 1)
        return counts

    # Velocity + novelty + merchant frequency per card
    df["txn_count_1h"], df["txn_count_24h"] = 0.0, 0.0
    df["time_since_last_sec"] = 0.0
    df["is_new_category_for_card"] = 0
    df["time_since_last_cat_sec"] = 0.0
    df["merchant_freq_ratio_card"] = 0.0
    df["merchant_rare_score_card"] = 0.0
    df.sort_values(["cc_num", "event_time"], inplace=True)
    last_ts_card: Dict[float, np.datetime64] = {}
    for card, g in df.groupby("cc_num", sort=False, dropna=False):
        idx = g.index
        df.loc[idx, "txn_count_1h"] = rolling_count_by_hours(g["event_time"], 1)
        df.loc[idx, "txn_count_24h"] = rolling_count_by_hours(g["event_time"], 24)
        # time since last per card
        last = None
        for ix, t in zip(g.index, g["event_time"].values):
            if last is not None and pd.notna(t):
                dt = (pd.Timestamp(t).to_datetime64() - pd.Timestamp(last).to_datetime64()).astype('timedelta64[s]').astype(float)
                df.loc[ix, "time_since_last_sec"] = dt
            else:
                df.loc[ix, "time_since_last_sec"] = np.nan
            last = t
        # per-card category novelty/time and merchant frequency
        seen_cat: Dict[str, bool] = {}
        last_cat_ts: Dict[str, np.datetime64] = {}
        merch_counts: Dict[str, int] = {}
        total_seen = 0
        for ix, cat, merch, t in zip(g.index, g["operation_type"].fillna("UNK").values, g["merchant_name"].fillna("UNK").values, g["event_time"].values):
            t64 = pd.Timestamp(t).to_datetime64() if pd.notna(t) else None
            df.loc[ix, "is_new_category_for_card"] = 0 if cat in seen_cat else 1
            if cat in last_cat_ts and t64 is not None:
                dt = (t64 - last_cat_ts[cat]).astype('timedelta64[s]').astype(float)
                df.loc[ix, "time_since_last_cat_sec"] = dt
            else:
                df.loc[ix, "time_since_last_cat_sec"] = np.nan
            seen_cat[cat] = True
            if t64 is not None:
                last_cat_ts[cat] = t64
            total_seen += 1
            mc = merch_counts.get(merch, 0)
            df.loc[ix, "merchant_freq_ratio_card"] = float(mc) / float(total_seen)
            df.loc[ix, "merchant_rare_score_card"] = 1.0 / float(mc + 1)
            merch_counts[merch] = mc + 1

    # Category winsor z_iqr
    cap = df.groupby("operation_type")["amount"].transform(lambda s: s.quantile(0.995))
    df["amount_winsor_cat"] = np.minimum(df["amount"], cap)
    cat_median = df.groupby("operation_type")["amount_winsor_cat"].transform("median")
    cat_iqr = df.groupby("operation_type")["amount_winsor_cat"].transform(lambda s: (s.quantile(0.75) - s.quantile(0.25)))
    df["amount_z_iqr_cat"] = (df["amount_winsor_cat"] - cat_median) / cat_iqr.replace({0: np.nan})

    # Feature blocks
    blocks: Dict[str, List[str]] = {
        "amount_core": ["amount", "abs_amount", "log1p_abs_amount"],
        "time": ["hour", "hour_sin", "hour_cos", "dow", "is_night"],
        "velocity": ["txn_count_1h", "txn_count_24h", "time_since_last_sec"],
        "cat_robust": ["amount_z_iqr_cat"],
        "novelty_cat": ["is_new_category_for_card", "time_since_last_cat_sec"],
        "merchant_freq": ["merchant_freq_ratio_card", "merchant_rare_score_card"],
    }

    categorical_features = ["operation_type", "merchant_name"]
    return df, blocks, categorical_features


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k_frac: float) -> float:
    n = len(scores)
    k = max(1, int(n * k_frac))
    order = np.argsort(scores)[::-1]
    topk = order[:k]
    return float(np.sum(y_true[topk] == 1) / k)


def run_config(df: pd.DataFrame, y: np.ndarray, num_cols: List[str], cat_cols: List[str], cfg: AblationConfig) -> Dict[str, float]:
    pre_num = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    pre_cat = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
    ])
    pre = ColumnTransformer(transformers=[
        ("num", pre_num, num_cols),
        ("cat", pre_cat, cat_cols),
    ])
    model = IsolationForest(
        n_estimators=cfg.n_estimators,
        max_samples="auto",
        max_features=cfg.max_features,
        contamination="auto",
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    pipe = Pipeline(steps=[("prep", pre), ("model", model)])

    # Fit on legitimate subset with contamination guard
    subset = df[y == 0].copy()
    p995 = float(np.percentile(subset["abs_amount"].dropna(), 99.5)) if subset["abs_amount"].notna().any() else np.inf
    subset = subset[subset["abs_amount"] < p995]
    pipe.fit(subset)

    decision = pipe.decision_function(df)
    anomaly_score = -decision
    ap = average_precision_score(y, anomaly_score)
    metrics = {
        "AP": float(ap),
        "P@0.1%": precision_at_k(y, anomaly_score, 0.001),
        "P@0.5%": precision_at_k(y, anomaly_score, 0.005),
        "P@1.0%": precision_at_k(y, anomaly_score, 0.01),
        "P@5.0%": precision_at_k(y, anomaly_score, 0.05),
    }
    return metrics


def main():
    cfg = AblationConfig()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ULB data...")
    df = load_ulb(cfg.row_limit)
    y = df["is_fraud"].astype(int).values
    print(f"Loaded {len(df):,} rows (fraud rate ~ {y.mean():.4%}). Engineering features...")

    df, blocks, cat_cols = engineer_features(df)

    # Define ablation configs
    configs = {
        "baseline_amount": ["amount_core"],
        "+time": ["amount_core", "time"],
        "+velocity": ["amount_core", "time", "velocity"],
        "+cat_robust": ["amount_core", "time", "velocity", "cat_robust"],
        "+novelty_cat": ["amount_core", "time", "velocity", "cat_robust", "novelty_cat"],
        "+merchant_freq": ["amount_core", "time", "velocity", "cat_robust", "novelty_cat", "merchant_freq"],
    }

    results = []
    for name, block_list in configs.items():
        num_cols = []
        for b in block_list:
            num_cols.extend(blocks[b])
        print(f"Running config: {name} with {len(num_cols)} numeric features")
        metrics = run_config(df, y, num_cols, cat_cols, cfg)
        results.append({"config": name, **metrics})

    out_csv = cfg.out_dir / "ulb_if_ablation.csv"
    out_json = cfg.out_dir / "ulb_if_ablation.json"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Wrote ablation results to {out_csv}")


if __name__ == "__main__":
    main()






