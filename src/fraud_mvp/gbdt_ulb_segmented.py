from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ULB_PATH = Path("/home/javier/repos/datasets/ULB Credit Card Fraud Dataset (European Cardholders)/fraudTrain.csv")


@dataclass
class Config:
    row_limit: int = 200_000
    random_state: int = 42
    out_dir: Path = Path("reports/phase2/ulb_gbdt")
    top_categories: int = 4


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

    # Amount core
    df["abs_amount"] = df["amount"].abs()
    df["log1p_abs_amount"] = np.log1p(df["abs_amount"].clip(lower=0))

    # Merchant sanitize
    df["merchant_name"] = (
        df["merchant_name"].astype(str).str.lower()
        .str.replace(r"^fraud_", "", regex=True)
        .str.replace(r"[^a-z0-9\s]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Category robust winsor z_iqr
    cap = df.groupby("operation_type")["amount"].transform(lambda s: s.quantile(0.995))
    df["amount_winsor_cat"] = np.minimum(df["amount"], cap)
    cat_median = df.groupby("operation_type")["amount_winsor_cat"].transform("median")
    cat_iqr = df.groupby("operation_type")["amount_winsor_cat"].transform(lambda s: (s.quantile(0.75) - s.quantile(0.25)))
    df["amount_z_iqr_cat"] = (df["amount_winsor_cat"] - cat_median) / cat_iqr.replace({0: np.nan})

    # Time
    df["hour"] = df["event_time"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow"] = df["event_time"].dt.dayofweek
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)

    # Per-card velocities and history (expanding mean/std via Welford)
    df["txn_count_24h"], df["time_since_last_sec"] = 0.0, 0.0
    df["card_mean_amt"], df["card_std_amt"], df["amt_z_card"] = 0.0, 0.0, 0.0
    for card, g in df.groupby("cc_num", sort=False):
        idx = g.index
        t_ns = g["event_time"].values.astype('datetime64[ns]').astype('int64')
        win = 24 * 3600 * 1_000_000_000
        counts = np.zeros(len(g), dtype=float)
        j = 0
        for i in range(len(t_ns)):
            while t_ns[i] - t_ns[j] > win and j < i:
                j += 1
            counts[i] = i - j + 1
        df.loc[idx, "txn_count_24h"] = counts
        # time since last + Welford
        mean, m2, n = 0.0, 0.0, 0
        last_t = None
        for k, (ix, amt, tt) in enumerate(zip(idx, g["abs_amount"].values, g["event_time"].values)):
            if last_t is not None:
                dt = (pd.Timestamp(tt).to_datetime64() - pd.Timestamp(last_t).to_datetime64()).astype('timedelta64[s]').astype(float)
                df.loc[ix, "time_since_last_sec"] = dt
            else:
                df.loc[ix, "time_since_last_sec"] = np.nan
            last_t = tt
            # Welford update
            n += 1
            delta = amt - mean
            mean += delta / n
            m2 += delta * (amt - mean)
            var = m2 / (n - 1) if n > 1 else 0.0
            std = np.sqrt(var)
            df.loc[ix, "card_mean_amt"] = mean
            df.loc[ix, "card_std_amt"] = std
            df.loc[ix, "amt_z_card"] = (amt - mean) / std if std > 0 else 0.0

    num_cols = [
        "amount", "abs_amount", "log1p_abs_amount", "amount_z_iqr_cat",
        "txn_count_24h", "time_since_last_sec",
        "card_mean_amt", "card_std_amt", "amt_z_card",
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


def build_gbdt_pipeline(num_cols: List[str], cat_cols: List[str], pos_weight: float, rs: int) -> Pipeline:
    pre = ColumnTransformer(transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline(steps=[
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False)),
        ]), cat_cols),
    ])
    clf = HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_depth=8,
        max_iter=300,
        random_state=rs,
        class_weight={0: 1.0, 1: float(pos_weight)},
    )
    return Pipeline(steps=[("prep", pre), ("gbdt", clf)])


def train_eval(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str], cfg: Config) -> Dict[str, float]:
    # chronological split
    df_sorted = df.sort_values("event_time")
    split_idx = int(len(df_sorted) * 0.8)
    train_idx, test_idx = df_sorted.index[:split_idx], df_sorted.index[split_idx:]
    X_train, X_test = df.loc[train_idx, :], df.loc[test_idx, :]
    y_train, y_test = df.loc[train_idx, "is_fraud"].astype(int).values, df.loc[test_idx, "is_fraud"].astype(int).values

    pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    pipe = build_gbdt_pipeline(num_cols, cat_cols, pos_weight, cfg.random_state)
    pipe.fit(X_train, y_train)

    # scores
    if hasattr(pipe.named_steps["gbdt"], "predict_proba"):
        scores = pipe.named_steps["gbdt"].predict_proba(pipe.named_steps["prep"].transform(X_test))[:, 1]
    else:
        scores = pipe.named_steps["gbdt"].predict(X_test)

    ap = average_precision_score(y_test, scores)
    return {
        "AP": float(ap),
        "P@0.5%": precision_at_k(y_test, scores, 0.005),
        "P@1.0%": precision_at_k(y_test, scores, 0.01),
        "P@5.0%": precision_at_k(y_test, scores, 0.05),
    }


def segmented_eval(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str], cfg: Config) -> Dict[str, float]:
    # chronological split
    df_sorted = df.sort_values("event_time")
    split_idx = int(len(df_sorted) * 0.8)
    train_df, test_df = df_sorted.iloc[:split_idx], df_sorted.iloc[split_idx:]

    # top categories by train volume
    top_cats = train_df["operation_type"].value_counts().head(cfg.top_categories).index.tolist()

    models: Dict[str, Pipeline] = {}
    pos_weights: Dict[str, float] = {}
    for cat in top_cats + ["__fallback"]:
        if cat == "__fallback":
            train_part = train_df[~train_df["operation_type"].isin(top_cats)]
        else:
            train_part = train_df[train_df["operation_type"] == cat]
        if len(train_part) < 1000:
            continue
        y_tr = train_part["is_fraud"].astype(int).values
        pw = (y_tr == 0).sum() / max(1, (y_tr == 1).sum())
        pos_weights[cat] = pw
        models[cat] = build_gbdt_pipeline(num_cols, cat_cols, pw, cfg.random_state)
        models[cat].fit(train_part, y_tr)

    # score test
    scores = np.zeros(len(test_df), dtype=float)
    for cat, part in test_df.groupby(test_df["operation_type"].map(lambda x: x if x in top_cats else "__fallback")):
        if cat not in models:
            continue
        idx = part.index
        pipe = models[cat]
        if hasattr(pipe.named_steps["gbdt"], "predict_proba"):
            sc = pipe.named_steps["gbdt"].predict_proba(pipe.named_steps["prep"].transform(part))[:, 1]
        else:
            sc = pipe.named_steps["gbdt"].predict(part)
        scores[test_df.index.get_indexer(idx)] = sc

    y_test = test_df["is_fraud"].astype(int).values
    ap = average_precision_score(y_test, scores)
    return {
        "AP": float(ap),
        "P@0.5%": precision_at_k(y_test, scores, 0.005),
        "P@1.0%": precision_at_k(y_test, scores, 0.01),
        "P@5.0%": precision_at_k(y_test, scores, 0.05),
    }


def main():
    cfg = Config()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_ulb(cfg.row_limit)
    df, num_cols, cat_cols = engineer(df)

    # Baseline single model with history
    base_metrics = train_eval(df, num_cols, cat_cols, cfg)

    # Segmented per category
    seg_metrics = segmented_eval(df, num_cols, cat_cols, cfg)

    # Save ablation
    results = [
        {"config": "gbdt_single+history", **base_metrics},
        {"config": "gbdt_segmented_by_category", **seg_metrics},
    ]
    out_csv = cfg.out_dir / "ulb_gbdt_ablation.csv"
    out_json = cfg.out_dir / "ulb_gbdt_ablation.json"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(results, indent=2))
    print("Wrote", out_csv)


if __name__ == "__main__":
    main()



