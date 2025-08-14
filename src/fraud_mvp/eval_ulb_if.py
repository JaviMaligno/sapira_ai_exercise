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
    n_estimators_grid: tuple[int, ...] = (200, 400, 600)
    max_features_grid: tuple[float, ...] = (0.5, 0.7, 1.0)
    max_samples_grid: tuple[object, ...] = ("auto", 256, 512)


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
    # Cyclic hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
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

    # Per-category winsorization at p99.5 and robust z-score using winsorized amount
    cap = df.groupby("operation_type")["amount"].transform(lambda s: s.quantile(0.995))
    df["amount_winsor_cat"] = np.minimum(df["amount"], cap)
    cat_median = df.groupby("operation_type")["amount_winsor_cat"].transform("median")
    cat_iqr = df.groupby("operation_type")["amount_winsor_cat"].transform(lambda s: (s.quantile(0.75) - s.quantile(0.25)))
    df["amount_z_iqr_cat"] = (df["amount_winsor_cat"] - cat_median) / cat_iqr.replace({0: np.nan})

    # Merchant text sanitization to avoid label leakage tokens like 'fraud_'
    if "merchant_name" in df.columns:
        s = df["merchant_name"].astype(str).str.lower()
        s = s.str.replace(r"^fraud_", "", regex=True)
        s = s.str.replace(r"[^a-z0-9\s]+", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
        df["merchant_name"] = s

    numeric_features = [
        "amount", "abs_amount", "log1p_abs_amount", "hour", "hour_sin", "hour_cos", "dow", "is_night",
        "txn_count_1h", "txn_count_24h", "time_since_last_sec", "amount_z_iqr_cat",
    ]
    categorical_features = ["operation_type", "merchant_name"]
    return df, numeric_features, categorical_features


def build_if_pipeline(numeric_features: List[str], categorical_features: List[str], random_state: int, n_estimators: int = 400, max_features: float | int = 1.0) -> Pipeline:
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
        n_estimators=n_estimators,
        max_samples="auto",
        max_features=max_features,
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
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

    print("Grid-searching Isolation Forest hyperparameters...")
    mask_legit = (y == 0)
    results = []
    best = None
    best_key = None
    for n_est in cfg.n_estimators_grid:
        for mf in cfg.max_features_grid:
            for ms in cfg.max_samples_grid:
                pipe = build_if_pipeline(num_cols, cat_cols, cfg.random_state, n_estimators=n_est, max_features=mf)
                subset = df.loc[mask_legit, :].copy()
                p995 = float(np.percentile(subset["abs_amount"].dropna(), 99.5)) if subset["abs_amount"].notna().any() else np.inf
                subset = subset[subset["abs_amount"] < p995]
                # Override max_samples if not auto
                pipe.named_steps["model"].set_params(max_samples=ms)
                pipe.fit(subset)

            decision = pipe.decision_function(df)
            anomaly_score = -decision
            ap = average_precision_score(y, anomaly_score)
            p_at_0_1 = precision_at_k(y, anomaly_score, 0.001)
            p_at_0_5 = precision_at_k(y, anomaly_score, 0.005)
            p_at_1_0 = precision_at_k(y, anomaly_score, 0.01)
            p_at_5_0 = precision_at_k(y, anomaly_score, 0.05)
            results.append({
                "n_estimators": n_est,
                "max_features": mf,
                    "max_samples": ms,
                "AP": float(ap),
                "P@0.1%": float(p_at_0_1),
                "P@0.5%": float(p_at_0_5),
                "P@1.0%": float(p_at_1_0),
                "P@5.0%": float(p_at_5_0),
            })
            key = (p_at_0_5, ap)
            if best is None or key > best_key:
                    best = (n_est, mf, ms, anomaly_score)
                    best_key = key

    # Save tuning table
    pd.DataFrame(results).to_csv(cfg.out_dir / "ulb_if_tuning.csv", index=False)
    n_best, mf_best, ms_best, anomaly_score = best

    # Save scores for best
    scores_out = cfg.out_dir / f"ulb_if_scores_n{n_best}.csv"
    pd.DataFrame({
        "transaction_id": df["transaction_id"],
        "is_fraud": y,
        "anomaly_score": anomaly_score,
    }).to_csv(scores_out, index=False)

    # Summaries across budgets
    ap = average_precision_score(y, anomaly_score)
    summary = {
        "rows": int(len(df)),
        "fraud_rate": float(y.mean()),
        "best_n_estimators": int(n_best),
        "best_max_features": float(mf_best),
        "average_precision": float(ap),
        "precision_at": {
            "0.1%": float(precision_at_k(y, anomaly_score, 0.001)),
            "0.5%": float(precision_at_k(y, anomaly_score, 0.005)),
            "1.0%": float(precision_at_k(y, anomaly_score, 0.01)),
            "5.0%": float(precision_at_k(y, anomaly_score, 0.05)),
        },
        "recommend_alert_rate": 0.005,
    }
    (cfg.out_dir / "ulb_if_eval_summary.json").write_text(json.dumps(summary, indent=2))
    (cfg.out_dir / "ulb_if_tuning_summary.json").write_text(json.dumps({"grid": results, "best": summary}, indent=2))
    # Derive per-category thresholds at recommended alert rate
    thresholds = {}
    if "operation_type" in df.columns:
        for cat, g in df.groupby("operation_type"):
            if len(g) >= 50:
                thr = float(np.quantile(anomaly_score[g.index], 1 - summary["recommend_alert_rate"]))
                thresholds[str(cat)] = thr
    (cfg.out_dir / "ulb_if_per_category_thresholds.json").write_text(json.dumps(thresholds, indent=2))

    print(f"Best n_estimators={n_best} max_features={mf_best} max_samples={ms_best} AP={ap:.4f} | P@0.5%={summary['precision_at']['0.5%']:.4f}")


if __name__ == "__main__":
    main()


