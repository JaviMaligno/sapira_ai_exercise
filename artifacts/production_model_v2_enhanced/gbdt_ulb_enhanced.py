from __future__ import annotations

import json
from collections import deque, defaultdict
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
from sklearn.isotonic import IsotonicRegression
import joblib
import shap


ULB_PATH = Path("/home/javier/repos/datasets/ULB Credit Card Fraud Dataset (European Cardholders)/fraudTrain.csv")
ULB_TEST_PATH = Path("/home/javier/repos/datasets/ULB Credit Card Fraud Dataset (European Cardholders)/fraudTest.csv")


@dataclass
class Config:
    row_limit: int = 200_000
    random_state: int = 42
    out_dir: Path = Path("reports/phase2/ulb_gbdt")
    # CV folds as chronological fractions (train_end, val_end)
    folds: List[Tuple[float, float]] = ((0.6, 0.8), (0.8, 1.0))
    per_category_alert_fracs: List[float] = (0.005, 0.01, 0.05)  # 0.5%, 1%, 5%
    export_alert_fracs: List[float] = (0.005, 0.01)


def load_ulb(limit: int) -> pd.DataFrame:
    usecols = ["trans_num", "unix_time", "category", "amt", "merchant", "is_fraud", "cc_num"]
    df = pd.read_csv(ULB_PATH, usecols=usecols, nrows=limit)
    df.rename(
        columns={
            "trans_num": "transaction_id",
            "unix_time": "event_time_ts",
            "category": "operation_type",
            "amt": "amount",
            "merchant": "merchant_name",
        },
        inplace=True,
    )
    df["event_time"] = pd.to_datetime(df["event_time_ts"], unit="s", utc=True)
    df["dataset"] = "ULB"
    return df


def load_ulb_test() -> pd.DataFrame:
    usecols = ["trans_num", "unix_time", "category", "amt", "merchant", "is_fraud", "cc_num"]
    df = pd.read_csv(ULB_TEST_PATH, usecols=usecols)
    df.rename(
        columns={
            "trans_num": "transaction_id",
            "unix_time": "event_time_ts",
            "category": "operation_type",
            "amt": "amount",
            "merchant": "merchant_name",
        },
        inplace=True,
    )
    df["event_time"] = pd.to_datetime(df["event_time_ts"], unit="s", utc=True)
    df["dataset"] = "ULB"
    return df


def load_ieee(limit: int) -> pd.DataFrame:
    tx_path = Path("/home/javier/repos/datasets/ieee-fraud-detection/train_transaction.csv")
    # Minimal columns used
    usecols = [
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        "ProductCD",
        "card1",
        "card4",
        "P_emaildomain",
        "isFraud",
    ]
    df = pd.read_csv(tx_path, usecols=usecols, nrows=limit)
    # Harmonize
    df.rename(
        columns={
            "TransactionID": "transaction_id",
            "TransactionDT": "event_time_ts",
            "TransactionAmt": "amount",
            "ProductCD": "operation_type",
            "isFraud": "is_fraud",
        },
        inplace=True,
    )
    # Build event_time from relative seconds since start
    base = pd.Timestamp("2017-12-01", tz="UTC")
    df["event_time"] = base + pd.to_timedelta(df["event_time_ts"], unit="s")
    # Merchant proxy
    merch = df["P_emaildomain"].fillna(df["card4"].astype(str)).fillna("UNK").astype(str)
    df["merchant_name"] = merch
    # Card proxy
    df["cc_num"] = df["card1"].fillna(-1).astype(int)
    df["dataset"] = "IEEE"
    return df[[
        "transaction_id", "event_time_ts", "event_time", "operation_type", "amount", "merchant_name", "is_fraud", "cc_num", "dataset"
    ]]


def engineer_enhanced(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = df.copy()
    df.sort_values("event_time", inplace=True)

    # Core amounts
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

    # Category-robust winsor z_iqr
    cap = df.groupby("operation_type")["amount"].transform(lambda s: s.quantile(0.995))
    df["amount_winsor_cat"] = np.minimum(df["amount"], cap)
    cat_median = df.groupby("operation_type")["amount_winsor_cat"].transform("median")
    cat_iqr = df.groupby("operation_type")["amount_winsor_cat"].transform(lambda s: (s.quantile(0.75) - s.quantile(0.25)))
    df["amount_z_iqr_cat"] = (df["amount_winsor_cat"] - cat_median) / cat_iqr.replace({0: np.nan})

    # Time features
    df["hour"] = df["event_time"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow"] = df["event_time"].dt.dayofweek
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)
    
    # Calendar seasonality features (recommended from evaluation)
    df["month"] = df["event_time"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # Initialize enhanced history/recurrence features
    df["txn_count_1h"] = 0.0
    df["txn_count_6h"] = 0.0
    df["txn_count_24h"] = 0.0
    df["amt_sum_24h"] = 0.0
    df["time_since_last_sec"] = 0.0
    df["is_new_merchant_for_card"] = 0
    df["unique_merchants_7d"] = 0.0
    df["unique_merchants_30d"] = 0.0
    df["prop_new_merchants_7d"] = 0.0
    df["prop_new_merchants_30d"] = 0.0
    df["merchant_count_7d"] = 0.0
    df["merchant_count_30d"] = 0.0
    df["days_since_last_merchant"] = np.nan
    
    # Balance velocity extensions (recommended from evaluation)
    df["amount_rolling_std_24h"] = 0.0
    df["amount_rolling_mean_24h"] = 0.0

    one_hour = np.timedelta64(3600, "s")
    six_hours = np.timedelta64(6 * 3600, "s")
    one_day = np.timedelta64(24 * 3600, "s")
    seven_days = np.timedelta64(7 * 24 * 3600, "s")
    thirty_days = np.timedelta64(30 * 24 * 3600, "s")

    # Process per card chronologically
    for card, g in df.groupby("cc_num", sort=False):
        idx = g.index
        times = g["event_time"].values.astype("datetime64[ns]")
        amounts = g["abs_amount"].values
        merchants = g["merchant_name"].fillna("UNK").values

        # Sliding windows
        win_1h = deque()
        win_6h = deque()
        win_24h = deque()  # items: (time, amount)
        last_time = None

        # Merchant tracking structures (point-in-time)
        first_seen_merchant: Dict[str, np.datetime64] = {}
        last_seen_merchant: Dict[str, np.datetime64] = {}
        # For 7/30d unique merchants: maintain count dicts
        merch_counts_7d: Dict[str, int] = defaultdict(int)
        merch_counts_30d: Dict[str, int] = defaultdict(int)
        # For proportion of new merchants: track recent flags
        flags_7d = deque()  # (time, is_new_flag)
        flags_30d = deque()
        # For merchant-specific counts over windows
        per_merchant_times_7d: Dict[str, deque] = defaultdict(deque)
        per_merchant_times_30d: Dict[str, deque] = defaultdict(deque)

        for i, (ix, t, a, m) in enumerate(zip(idx, times, amounts, merchants)):
            # Pop old entries from sliding windows
            def prune(d: deque, window: np.timedelta64):
                while d and (t - d[0] > window):
                    d.popleft()

            prune(win_1h, one_hour)
            prune(win_6h, six_hours)
            while win_24h and (t - win_24h[0][0] > one_day):
                win_24h.popleft()

            # Push current
            win_1h.append(t)
            win_6h.append(t)
            win_24h.append((t, a))

            # Counts and sums
            df.loc[ix, "txn_count_1h"] = float(len(win_1h))
            df.loc[ix, "txn_count_6h"] = float(len(win_6h))
            df.loc[ix, "txn_count_24h"] = float(len(win_24h))
            df.loc[ix, "amt_sum_24h"] = float(sum(v for _, v in win_24h))
            
            # Balance velocity extensions: rolling statistics of amounts
            if len(win_24h) >= 2:
                amounts_24h = [v for _, v in win_24h]
                df.loc[ix, "amount_rolling_std_24h"] = float(np.std(amounts_24h))
                df.loc[ix, "amount_rolling_mean_24h"] = float(np.mean(amounts_24h))
            else:
                # Single transaction or no history
                df.loc[ix, "amount_rolling_std_24h"] = 0.0
                df.loc[ix, "amount_rolling_mean_24h"] = float(a)

            # Time since last
            if last_time is None:
                df.loc[ix, "time_since_last_sec"] = np.nan
            else:
                dt = (t - last_time).astype("timedelta64[s]").astype(float)
                df.loc[ix, "time_since_last_sec"] = dt
            last_time = t

            # Merchant recurrence flags
            is_new = 1 if m not in first_seen_merchant else 0
            if is_new:
                first_seen_merchant[m] = t
            df.loc[ix, "is_new_merchant_for_card"] = is_new

            # Days since last seen merchant
            last_m = last_seen_merchant.get(m)
            if last_m is None:
                df.loc[ix, "days_since_last_merchant"] = np.nan
            else:
                df.loc[ix, "days_since_last_merchant"] = (t - last_m).astype("timedelta64[D]").astype(float)
            last_seen_merchant[m] = t

            # Maintain 7d/30d merchant windows
            # Evict old from count dicts using per-merchant timestamp queues
            def update_window(per_times: Dict[str, deque], counts: Dict[str, int], window: np.timedelta64):
                # add current
                q = per_times[m]
                q.append(t)
                counts[m] += 1
                # prune older for all merchants
                for merch_key in list(per_times.keys()):
                    qk = per_times[merch_key]
                    while qk and (t - qk[0] > window):
                        qk.popleft()
                        counts[merch_key] -= 1
                        if counts[merch_key] <= 0:
                            del counts[merch_key]
                            del per_times[merch_key]

            update_window(per_merchant_times_7d, merch_counts_7d, seven_days)
            update_window(per_merchant_times_30d, merch_counts_30d, thirty_days)

            df.loc[ix, "unique_merchants_7d"] = float(len(merch_counts_7d))
            df.loc[ix, "unique_merchants_30d"] = float(len(merch_counts_30d))

            # Proportion of new merchants in window: maintain flag deques
            def update_prop(flags: deque, window: np.timedelta64, flag_value: int) -> float:
                flags.append((t, flag_value))
                while flags and (t - flags[0][0] > window):
                    flags.popleft()
                if not flags:
                    return 0.0
                s = sum(v for _, v in flags)
                return float(s / len(flags))

            df.loc[ix, "prop_new_merchants_7d"] = update_prop(flags_7d, seven_days, is_new)
            df.loc[ix, "prop_new_merchants_30d"] = update_prop(flags_30d, thirty_days, is_new)

            # Counts to current merchant within 7/30d
            df.loc[ix, "merchant_count_7d"] = float(len(per_merchant_times_7d.get(m, ())))
            df.loc[ix, "merchant_count_30d"] = float(len(per_merchant_times_30d.get(m, ())))

    # Global merchant frequency rank (train-like map will be used in train function)
    # Placeholder column that will be filled with train-only stats later
    df["merchant_freq_global"] = 0.0

    num_cols = [
        "amount",
        "abs_amount",
        "log1p_abs_amount",
        "amount_z_iqr_cat",
        "txn_count_1h",
        "txn_count_6h",
        "txn_count_24h",
        "amt_sum_24h",
        "time_since_last_sec",
        "is_new_merchant_for_card",
        "unique_merchants_7d",
        "unique_merchants_30d",
        "prop_new_merchants_7d",
        "prop_new_merchants_30d",
        "merchant_count_7d",
        "merchant_count_30d",
        "days_since_last_merchant",
        "hour",
        "hour_sin",
        "hour_cos",
        "dow",
        "is_night",
        "month_sin",
        "month_cos",
        "amount_rolling_std_24h",
        "amount_rolling_mean_24h",
        "merchant_freq_global",
    ]
    cat_cols = ["operation_type", "dataset"]  # merchant_name via frequency; add dataset indicator
    return df, num_cols, cat_cols


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k_frac: float) -> float:
    n = len(scores)
    k = max(1, int(n * k_frac))
    order = np.argsort(scores)[::-1]
    topk = order[:k]
    return float(np.sum(y_true[topk] == 1) / k)


def build_pipeline(num_cols: List[str], cat_cols: List[str], pos_weight: float, rs: int) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        (
                            "oh",
                            OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ]
    )
    clf = HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_depth=8,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        l2_regularization=1e-3,
        random_state=rs,
        class_weight={0: 1.0, 1: float(pos_weight)},
    )
    return Pipeline(steps=[("prep", pre), ("gbdt", clf)])


def compute_scores(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(pipe.named_steps["gbdt"], "predict_proba"):
        return pipe.named_steps["gbdt"].predict_proba(pipe.named_steps["prep"].transform(X))[:, 1]
    return pipe.named_steps["gbdt"].predict(X)


def fit_if_and_rule_scores(X_train: pd.DataFrame, X_valid: pd.DataFrame, num_cols: List[str], cat_cols_for_if: List[str], rs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Isolation Forest stack
    if_prep = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        (
                            "oh",
                            OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False),
                        ),
                    ]
                ),
                cat_cols_for_if,
            ),
        ]
    )
    if_model = IsolationForest(n_estimators=200, max_samples="auto", contamination="auto", random_state=rs, n_jobs=-1)
    if_pipe = Pipeline(steps=[("prep", if_prep), ("if", if_model)])
    X_train_legit = X_train[X_train["is_fraud"].astype(int).values == 0]
    if_pipe.fit(X_train_legit)
    train_if_score = -if_pipe.named_steps["if"].decision_function(if_pipe.named_steps["prep"].transform(X_train))
    valid_if_score = -if_pipe.named_steps["if"].decision_function(if_pipe.named_steps["prep"].transform(X_valid))

    # Rule score based on train-derived quantiles
    p995 = float(np.nanpercentile(X_train["abs_amount"], 99.5)) if X_train["abs_amount"].notna().any() else np.nan

    def compute_rule_score(dfp: pd.DataFrame) -> np.ndarray:
        high_amt = ((dfp["abs_amount"] >= p995) & dfp["abs_amount"].notna()).astype(int)
        new_merch_high = ((dfp["is_new_merchant_for_card"] == 1) & (dfp["abs_amount"] >= p995)).astype(int)
        rapid = (dfp["txn_count_24h"] >= 10).astype(int)
        return (1.5 * high_amt + 1.0 * new_merch_high + 0.5 * rapid).values

    train_rule = compute_rule_score(X_train)
    valid_rule = compute_rule_score(X_valid)
    return train_if_score, valid_if_score, train_rule, valid_rule


def time_aware_cv(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str], cfg: Config) -> Dict:
    df_sorted = df.sort_values("event_time")
    y = df_sorted["is_fraud"].astype(int).values

    # Global merchant frequency mapping computed on the earliest training window (per fold)
    results = []
    for train_frac, val_frac in cfg.folds:
        train_end = int(len(df_sorted) * train_frac)
        val_end = int(len(df_sorted) * val_frac)
        train_df = df_sorted.iloc[:train_end].copy()
        val_df = df_sorted.iloc[train_end:val_end].copy()

        # Merchant frequency map on train only
        freq_map = train_df["merchant_name"].fillna("UNK").value_counts(normalize=True).to_dict()
        train_df["merchant_freq_global"] = train_df["merchant_name"].fillna("UNK").map(freq_map).fillna(0.0)
        val_df["merchant_freq_global"] = val_df["merchant_name"].fillna("UNK").map(freq_map).fillna(0.0)

        # Stacked features (IF + rule)
        train_if, val_if, train_rule, val_rule = fit_if_and_rule_scores(
            train_df, val_df, num_cols, cat_cols_for_if=["operation_type", "merchant_name"], rs=cfg.random_state
        )
        train_df_ext = train_df.copy()
        val_df_ext = val_df.copy()
        train_df_ext["if_anomaly_score"], val_df_ext["if_anomaly_score"] = train_if, val_if
        train_df_ext["rule_score"], val_df_ext["rule_score"] = train_rule, val_rule

        num_cols_ext = num_cols + ["if_anomaly_score", "rule_score"]
        pos_weight = (train_df_ext["is_fraud"].astype(int).values == 0).sum() / max(1, (train_df_ext["is_fraud"].astype(int).values == 1).sum())
        pipe = build_pipeline(num_cols_ext, cat_cols, pos_weight, cfg.random_state)
        pipe.fit(train_df_ext, train_df_ext["is_fraud"].astype(int).values)
        scores = compute_scores(pipe, val_df_ext)
        y_val = val_df_ext["is_fraud"].astype(int).values
        ap = average_precision_score(y_val, scores)
        fold_metrics = {
            "train_frac": float(train_frac),
            "val_frac": float(val_frac),
            "rows_train": int(len(train_df_ext)),
            "rows_val": int(len(val_df_ext)),
            "AP": float(ap),
            "P@0.5%": precision_at_k(y_val, scores, 0.005),
            "P@1.0%": precision_at_k(y_val, scores, 0.01),
            "P@5.0%": precision_at_k(y_val, scores, 0.05),
        }
        results.append(fold_metrics)

    return {"folds": results}


def time_aware_hp_tuning(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str], cfg: Config) -> Dict:
    df_sorted = df.sort_values("event_time")
    results = []
    # Small grid
    grid = [
        {"max_depth": d, "min_samples_leaf": msl, "learning_rate": lr, "l2_regularization": l2}
        for d in (6, 8)
        for msl in (20, 50)
        for lr in (0.1,)
        for l2 in (1e-3, 3e-3)
    ]

    for params in grid:
        fold_scores = []
        for train_frac, val_frac in cfg.folds:
            train_end = int(len(df_sorted) * train_frac)
            val_end = int(len(df_sorted) * val_frac)
            train_df = df_sorted.iloc[:train_end].copy()
            val_df = df_sorted.iloc[train_end:val_end].copy()

            # Merchant frequency map on train only
            freq_map = train_df["merchant_name"].fillna("UNK").value_counts(normalize=True).to_dict()
            train_df["merchant_freq_global"] = train_df["merchant_name"].fillna("UNK").map(freq_map).fillna(0.0)
            val_df["merchant_freq_global"] = val_df["merchant_name"].fillna("UNK").map(freq_map).fillna(0.0)

            # Stacked features (IF + rule)
            train_if, val_if, train_rule, val_rule = fit_if_and_rule_scores(
                train_df, val_df, num_cols, cat_cols_for_if=["operation_type", "merchant_name"], rs=cfg.random_state
            )
            train_df_ext = train_df.copy()
            val_df_ext = val_df.copy()
            train_df_ext["if_anomaly_score"], val_df_ext["if_anomaly_score"] = train_if, val_if
            train_df_ext["rule_score"], val_df_ext["rule_score"] = train_rule, val_rule

            num_cols_ext = num_cols + ["if_anomaly_score", "rule_score"]
            pos_weight = (train_df_ext["is_fraud"].astype(int).values == 0).sum() / max(1, (train_df_ext["is_fraud"].astype(int).values == 1).sum())

            # Build model with params
            pre = ColumnTransformer(
                transformers=[
                    ("num", SimpleImputer(strategy="median"), num_cols_ext),
                    (
                        "cat",
                        Pipeline(
                            steps=[
                                ("imp", SimpleImputer(strategy="most_frequent")),
                                ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False)),
                            ]
                        ),
                        cat_cols,
                    ),
                ]
            )
            clf = HistGradientBoostingClassifier(
                learning_rate=params["learning_rate"],
                max_depth=params["max_depth"],
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                l2_regularization=params["l2_regularization"],
                max_bins=255,
                random_state=cfg.random_state,
                class_weight={0: 1.0, 1: float(pos_weight)},
            )
            pipe = Pipeline(steps=[("prep", pre), ("gbdt", clf)])
            pipe.fit(train_df_ext, train_df_ext["is_fraud"].astype(int).values)
            scores = compute_scores(pipe, val_df_ext)
            y_val = val_df_ext["is_fraud"].astype(int).values
            ap = average_precision_score(y_val, scores)
            fold_scores.append(ap)

        results.append({"params": params, "AP_mean": float(np.mean(fold_scores)), "AP_folds": [float(x) for x in fold_scores]})

    # Sort by AP_mean desc
    results_sorted = sorted(results, key=lambda x: x["AP_mean"], reverse=True)
    best = results_sorted[0] if results_sorted else {}
    return {"grid_results": results_sorted, "best": best}


def per_category_thresholds(scores: np.ndarray, y_true: np.ndarray, categories: np.ndarray, alert_frac: float) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    for cat in np.unique(categories):
        mask = categories == cat
        sc = scores[mask]
        if len(sc) == 0:
            continue
        k = max(1, int(len(sc) * alert_frac))
        thr = float(np.partition(sc, -k)[-k])  # kth largest value
        thresholds[str(cat)] = thr
    return thresholds


def evaluate_with_thresholds(scores: np.ndarray, y_true: np.ndarray, categories: np.ndarray, thresholds: Dict[str, float]) -> Dict[str, float]:
    flags = np.array([scores[i] >= thresholds.get(str(categories[i]), np.inf) for i in range(len(scores))])
    # Precision among flagged
    if flags.sum() == 0:
        precision = 0.0
    else:
        precision = float((y_true[flags] == 1).mean())
    return {"alerts": int(flags.sum()), "precision": precision}


def train_eval_and_calibrate(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str], cfg: Config) -> Dict:
    df_sorted = df.sort_values("event_time")
    split_idx = int(len(df_sorted) * 0.8)
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()

    # Merchant global frequency (train only)
    freq_map = train_df["merchant_name"].fillna("UNK").value_counts(normalize=True).to_dict()
    train_df["merchant_freq_global"] = train_df["merchant_name"].fillna("UNK").map(freq_map).fillna(0.0)
    test_df["merchant_freq_global"] = test_df["merchant_name"].fillna("UNK").map(freq_map).fillna(0.0)

    # Stacked features (IF + rule)
    train_if, test_if, train_rule, test_rule = fit_if_and_rule_scores(
        train_df, test_df, num_cols, cat_cols_for_if=["operation_type", "merchant_name"], rs=cfg.random_state
    )
    train_df_ext = train_df.copy()
    test_df_ext = test_df.copy()
    train_df_ext["if_anomaly_score"], test_df_ext["if_anomaly_score"] = train_if, test_if
    train_df_ext["rule_score"], test_df_ext["rule_score"] = train_rule, test_rule

    num_cols_ext = num_cols + ["if_anomaly_score", "rule_score"]
    pos_weight = (train_df_ext["is_fraud"].astype(int).values == 0).sum() / max(1, (train_df_ext["is_fraud"].astype(int).values == 1).sum())
    pipe = build_pipeline(num_cols_ext, cat_cols, pos_weight, cfg.random_state)
    # Hold out last 10% of train for isotonic calibration
    calib_split = int(len(train_df_ext) * 0.9)
    tr_cal = train_df_ext.iloc[:calib_split]
    te_cal = train_df_ext.iloc[calib_split:]
    y_tr_cal = tr_cal["is_fraud"].astype(int).values
    y_te_cal = te_cal["is_fraud"].astype(int).values
    pipe.fit(tr_cal, y_tr_cal)
    # Raw scores
    pre: ColumnTransformer = pipe.named_steps["prep"]
    scores_cal_raw = compute_scores(pipe, te_cal)
    # Fit isotonic on calibration slice
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(scores_cal_raw, y_te_cal)
    # Test scores calibrated
    scores_raw = compute_scores(pipe, test_df_ext)
    scores = iso.transform(scores_raw)
    y_test = test_df_ext["is_fraud"].astype(int).values

    # Overall metrics
    ap = average_precision_score(y_test, scores)
    overall = {
        "rows": int(len(df_sorted)),
        "train_rows": int(len(train_df_ext)),
        "test_rows": int(len(test_df_ext)),
        "fraud_rate_train": float((train_df_ext["is_fraud"].astype(int).values == 1).mean()),
        "fraud_rate_test": float((y_test == 1).mean()),
        "average_precision": float(ap),
        "precision_at": {
            "0.5%": precision_at_k(y_test, scores, 0.005),
            "1.0%": precision_at_k(y_test, scores, 0.01),
            "5.0%": precision_at_k(y_test, scores, 0.05),
        },
    }

    # Per-category thresholds calibration
    thresholds_by_frac: Dict[str, Dict[str, float]] = {}
    per_frac_eval: Dict[str, Dict[str, float]] = {}
    cats = test_df_ext["operation_type"].astype(str).values
    for frac in cfg.per_category_alert_fracs:
        thrs = per_category_thresholds(scores, y_test, cats, frac)
        thresholds_by_frac[str(frac)] = thrs
        per_frac_eval[str(frac)] = evaluate_with_thresholds(scores, y_test, cats, thrs)

    # SHAP explanations (in-period test)
    X_test_mat = pre.transform(test_df_ext)
    try:
        explainer = shap.Explainer(pipe.named_steps["gbdt"])
        # Global: sample subset for efficiency
        rng = np.random.default_rng(cfg.random_state)
        sample_size = min(2000, X_test_mat.shape[0])
        sample_idx = rng.choice(X_test_mat.shape[0], size=sample_size, replace=False)
        shap_vals_sample = explainer(X_test_mat[sample_idx])
        # Feature names
        raw_names = pre.get_feature_names_out()
        def clean_name(n: str) -> str:
            n = n.replace("num__", "")
            n = n.replace("cat__oh__", "")
            n = n.replace("cat__", "")
            return n
        feat_names = [clean_name(n) for n in raw_names]
        mean_abs = np.mean(np.abs(shap_vals_sample.values), axis=0)
        order = np.argsort(mean_abs)[::-1]
        global_importance = [
            {"feature": feat_names[i], "mean_abs_shap": float(mean_abs[i])}
            for i in order
        ]
        # Per-alert: top 100 highest scores
        top_n = min(100, X_test_mat.shape[0])
        top_idx = np.argsort(scores)[-top_n:][::-1]
        shap_vals_top = explainer(X_test_mat[top_idx])
        top_alerts = []
        for rank, idx in enumerate(top_idx):
            vals = shap_vals_top.values[rank]
            contrib_order = np.argsort(np.abs(vals))[::-1][:5]
            contribs = [
                {"feature": feat_names[j], "shap_value": float(vals[j])}
                for j in contrib_order
            ]
            row = test_df_ext.iloc[idx]
            top_alerts.append({
                "rank": int(rank + 1),
                "score": float(scores[idx]),
                "event_time": str(row.get("event_time")),
                "operation_type": str(row.get("operation_type")),
                "amount": float(row.get("amount", np.nan)),
                "merchant_name": str(row.get("merchant_name")),
                "reasons": contribs,
            })
        shap_artifacts = {
            "global_importance": global_importance,
            "top_alerts": top_alerts,
        }
    except Exception as e:
        shap_artifacts = {"error": str(e)}

    # Persist model artifacts for serving
    # 1) Fitted pipeline and isotonic calibrator
    joblib.dump(pipe, cfg.out_dir / "pipeline.pkl")
    joblib.dump(iso, cfg.out_dir / "isotonic.pkl")
    # 2) Isolation Forest pipeline for stacked feature at serve time (fit on train legitimate)
    if_prep = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False)),
                    ]
                ),
                ["operation_type", "merchant_name"],
            ),
        ]
    )
    if_model = IsolationForest(n_estimators=200, max_samples="auto", contamination="auto", random_state=cfg.random_state, n_jobs=-1)
    if_pipe = Pipeline(steps=[("prep", if_prep), ("if", if_model)])
    X_train_legit = train_df[train_df["is_fraud"].astype(int).values == 0]
    if_pipe.fit(X_train_legit)
    joblib.dump(if_pipe, cfg.out_dir / "if_pipe.pkl")
    # 3) Merchant frequency map and rule parameters
    (cfg.out_dir / "merchant_freq_map.json").write_text(json.dumps(freq_map, indent=2))
    p995 = float(np.nanpercentile(train_df["abs_amount"], 99.5)) if train_df["abs_amount"].notna().any() else float("nan")
    (cfg.out_dir / "rule_params.json").write_text(json.dumps({"p995": p995}, indent=2))

    return {
        "overall": overall,
        "thresholds": thresholds_by_frac,
        "threshold_eval": per_frac_eval,
        "shap": shap_artifacts,
    }


def evaluate_on_ulb_test(df_train_full: pd.DataFrame, num_cols: List[str], cat_cols: List[str], cfg: Config) -> Dict:
    # Train on 80% of fraudTrain, evaluate on fraudTest (later period)
    df_train_full = df_train_full.sort_values("event_time")
    split_idx = int(len(df_train_full) * 0.8)
    train_df = df_train_full.iloc[:split_idx].copy()
    ulb_test_raw = load_ulb_test()
    ulb_test_df, _, _ = engineer_enhanced(ulb_test_raw)

    # Merchant global frequency (train only)
    freq_map = train_df["merchant_name"].fillna("UNK").value_counts(normalize=True).to_dict()
    train_df["merchant_freq_global"] = train_df["merchant_name"].fillna("UNK").map(freq_map).fillna(0.0)
    ulb_test_df["merchant_freq_global"] = ulb_test_df["merchant_name"].fillna("UNK").map(freq_map).fillna(0.0)

    # Stacked features (IF + rule) fit on train only
    train_if, test_if, train_rule, test_rule = fit_if_and_rule_scores(
        train_df, ulb_test_df, num_cols, cat_cols_for_if=["operation_type", "merchant_name"], rs=cfg.random_state
    )
    train_df_ext = train_df.copy()
    test_df_ext = ulb_test_df.copy()
    train_df_ext["if_anomaly_score"], test_df_ext["if_anomaly_score"] = train_if, test_if
    train_df_ext["rule_score"], test_df_ext["rule_score"] = train_rule, test_rule

    num_cols_ext = num_cols + ["if_anomaly_score", "rule_score"]
    pos_weight = (train_df_ext["is_fraud"].astype(int).values == 0).sum() / max(1, (train_df_ext["is_fraud"].astype(int).values == 1).sum())
    pipe = build_pipeline(num_cols_ext, cat_cols, pos_weight, cfg.random_state)
    pipe.fit(train_df_ext, train_df_ext["is_fraud"].astype(int).values)
    scores = compute_scores(pipe, test_df_ext)
    y_test = test_df_ext["is_fraud"].astype(int).values

    ap = average_precision_score(y_test, scores)
    overall = {
        "train_rows": int(len(train_df_ext)),
        "test_rows": int(len(test_df_ext)),
        "fraud_rate_train": float((train_df_ext["is_fraud"].astype(int).values == 1).mean()),
        "fraud_rate_test": float((y_test == 1).mean()),
        "average_precision": float(ap),
        "precision_at": {
            "0.5%": precision_at_k(y_test, scores, 0.005),
            "1.0%": precision_at_k(y_test, scores, 0.01),
            "5.0%": precision_at_k(y_test, scores, 0.05),
        },
    }

    thresholds_by_frac: Dict[str, Dict[str, float]] = {}
    per_frac_eval: Dict[str, Dict[str, float]] = {}
    cats = test_df_ext["operation_type"].astype(str).values
    for frac in cfg.per_category_alert_fracs:
        thrs = per_category_thresholds(scores, y_test, cats, frac)
        thresholds_by_frac[str(frac)] = thrs
        per_frac_eval[str(frac)] = evaluate_with_thresholds(scores, y_test, cats, thrs)

    return {"overall": overall, "thresholds": thresholds_by_frac, "threshold_eval": per_frac_eval}


def train_eval_on_combined_ulb_ieee(limit_ulb: int, limit_ieee: int, cfg: Config) -> Dict:
    # Load both datasets
    ulb_raw = load_ulb(limit_ulb)
    ieee_raw = load_ieee(limit_ieee)
    df_all = pd.concat([ulb_raw, ieee_raw], ignore_index=True)
    df_all, num_cols, cat_cols = engineer_enhanced(df_all)

    # Chronological split across combined
    df_sorted = df_all.sort_values("event_time")
    split_idx = int(len(df_sorted) * 0.8)
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()

    # Train-only merchant frequency
    freq_map = train_df["merchant_name"].fillna("UNK").value_counts(normalize=True).to_dict()
    train_df["merchant_freq_global"] = train_df["merchant_name"].fillna("UNK").map(freq_map).fillna(0.0)
    test_df["merchant_freq_global"] = test_df["merchant_name"].fillna("UNK").map(freq_map).fillna(0.0)

    # Stacked IF + rule
    train_if, test_if, train_rule, test_rule = fit_if_and_rule_scores(
        train_df, test_df, num_cols, cat_cols_for_if=["operation_type", "merchant_name", "dataset"], rs=cfg.random_state
    )
    train_df_ext = train_df.copy()
    test_df_ext = test_df.copy()
    train_df_ext["if_anomaly_score"], test_df_ext["if_anomaly_score"] = train_if, test_if
    train_df_ext["rule_score"], test_df_ext["rule_score"] = train_rule, test_rule

    num_cols_ext = num_cols + ["if_anomaly_score", "rule_score"]
    # Use dataset in categoricals
    cat_cols_ext = ["operation_type", "dataset"]
    pos_weight = (train_df_ext["is_fraud"].astype(int).values == 0).sum() / max(1, (train_df_ext["is_fraud"].astype(int).values == 1).sum())
    pipe = build_pipeline(num_cols_ext, cat_cols_ext, pos_weight, cfg.random_state)

    # Isotonic calibration on last 10% of train
    calib_split = int(len(train_df_ext) * 0.9)
    tr_cal = train_df_ext.iloc[:calib_split]
    te_cal = train_df_ext.iloc[calib_split:]
    y_tr_cal = tr_cal["is_fraud"].astype(int).values
    y_te_cal = te_cal["is_fraud"].astype(int).values
    pipe.fit(tr_cal, y_tr_cal)
    scores_cal_raw = compute_scores(pipe, te_cal)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(scores_cal_raw, y_te_cal)

    # Test scores
    scores_raw = compute_scores(pipe, test_df_ext)
    scores = iso.transform(scores_raw)
    y_test = test_df_ext["is_fraud"].astype(int).values

    # Overall metrics
    ap = average_precision_score(y_test, scores)
    overall = {
        "rows": int(len(df_sorted)),
        "train_rows": int(len(train_df_ext)),
        "test_rows": int(len(test_df_ext)),
        "fraud_rate_train": float((train_df_ext["is_fraud"].astype(int).values == 1).mean()),
        "fraud_rate_test": float((y_test == 1).mean()),
        "average_precision": float(ap),
        "precision_at": {
            "0.5%": precision_at_k(y_test, scores, 0.005),
            "1.0%": precision_at_k(y_test, scores, 0.01),
            "5.0%": precision_at_k(y_test, scores, 0.05),
        },
    }

    # Per-dataset slice metrics
    per_dataset = {}
    for ds in ["ULB", "IEEE"]:
        mask = (test_df_ext["dataset"].astype(str).values == ds)
        if mask.sum() == 0:
            continue
        y_ds = y_test[mask]
        s_ds = scores[mask]
        per_dataset[ds] = {
            "rows": int(mask.sum()),
            "average_precision": float(average_precision_score(y_ds, s_ds)),
            "precision_at": {
                "0.5%": precision_at_k(y_ds, s_ds, 0.005),
                "1.0%": precision_at_k(y_ds, s_ds, 0.01),
                "5.0%": precision_at_k(y_ds, s_ds, 0.05),
            },
        }

    # Per-dataset thresholds export (per-category within dataset)
    thresholds_by_dataset: Dict[str, Dict[str, Dict[str, float]]] = {}
    for frac in cfg.export_alert_fracs:
        key = str(frac)
        thresholds_by_dataset[key] = {}
        for ds in ["ULB", "IEEE"]:
            mask = (test_df_ext["dataset"].astype(str).values == ds)
            if mask.sum() == 0:
                continue
            thrs = per_category_thresholds(scores[mask], y_test[mask], test_df_ext.loc[mask, "operation_type"].astype(str).values, frac)
            thresholds_by_dataset[key][ds] = thrs

    return {"overall": overall, "per_dataset": per_dataset, "thresholds_by_dataset": thresholds_by_dataset}


def main():
    cfg = Config()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # Load and engineer
    df = load_ulb(cfg.row_limit)
    df, num_cols, cat_cols = engineer_enhanced(df)

    # CV
    cv_summary = time_aware_cv(df, num_cols, cat_cols, cfg)
    (cfg.out_dir / "ulb_gbdt_cv.json").write_text(json.dumps(cv_summary, indent=2))
    print("Wrote", cfg.out_dir / "ulb_gbdt_cv.json")

    # Train, test, calibrate
    final_summary = train_eval_and_calibrate(df, num_cols, cat_cols, cfg)
    (cfg.out_dir / "ulb_gbdt_enhanced_summary.json").write_text(json.dumps(final_summary, indent=2))
    print("Wrote", cfg.out_dir / "ulb_gbdt_enhanced_summary.json")

    # Export threshold files for serving (per-category)
    for frac in cfg.export_alert_fracs:
        key = str(frac)
        thr_map = final_summary.get("thresholds", {}).get(key, {})
        if thr_map:
            out_path = cfg.out_dir / f"gbdt_per_category_thresholds_{str(frac).replace('.', 'p')}.json"
            out_path.write_text(json.dumps(thr_map, indent=2))
            print("Wrote", out_path)

    # Export SHAP artifacts as standalone for analysts
    shap_block = final_summary.get("shap", {})
    if shap_block and isinstance(shap_block, dict):
        (cfg.out_dir / "ulb_gbdt_shap_global.json").write_text(json.dumps(shap_block.get("global_importance", []), indent=2))
        (cfg.out_dir / "ulb_gbdt_shap_top_alerts.json").write_text(json.dumps(shap_block.get("top_alerts", []), indent=2))
        print("Wrote", cfg.out_dir / "ulb_gbdt_shap_global.json")
        print("Wrote", cfg.out_dir / "ulb_gbdt_shap_top_alerts.json")

    # Evaluate on ULB fraudTest as later-time holdout
    ulbtest_summary = evaluate_on_ulb_test(df, num_cols, cat_cols, cfg)
    (cfg.out_dir / "ulb_gbdt_enhanced_ulbtest_summary.json").write_text(json.dumps(ulbtest_summary, indent=2))
    print("Wrote", cfg.out_dir / "ulb_gbdt_enhanced_ulbtest_summary.json")

    # Time-aware HP tuning (small grid)
    tuning = time_aware_hp_tuning(df, num_cols, cat_cols, cfg)
    (cfg.out_dir / "ulb_gbdt_tuning.json").write_text(json.dumps(tuning, indent=2))
    print("Wrote", cfg.out_dir / "ulb_gbdt_tuning.json")

    # Combined ULB + IEEE evaluation
    combined = train_eval_on_combined_ulb_ieee(cfg.row_limit, 300_000, cfg)
    (cfg.out_dir / "ulb_ieee_gbdt_enhanced_summary.json").write_text(json.dumps(combined, indent=2))
    print("Wrote", cfg.out_dir / "ulb_ieee_gbdt_enhanced_summary.json")

    # Export per-dataset thresholds for serving
    thrs_ds = combined.get("thresholds_by_dataset", {})
    for frac_key, ds_map in thrs_ds.items():
        for ds, thrs in ds_map.items():
            out_path = cfg.out_dir / f"gbdt_thresholds_{ds}_{str(frac_key).replace('.', 'p')}.json"
            out_path.write_text(json.dumps(thrs, indent=2))
            print("Wrote", out_path)

    # Versioned serving config (example v1) using ULB 0.5% thresholds
    serving_cfg = {
        "model": {
            "type": "HistGradientBoostingClassifier",
            "hyperparams": {"max_depth": 8, "learning_rate": 0.1, "l2_regularization": 0.003},
            "calibration": "isotonic",
        },
        "thresholds": {
            "ulb_per_category@0.5%": f"{cfg.out_dir}/gbdt_per_category_thresholds_0p005.json",
            "ieee_per_category@0.5%": f"{cfg.out_dir}/gbdt_thresholds_IEEE_0p005.json",
        },
        "artifacts": {
            "summary": f"{cfg.out_dir}/ulb_gbdt_enhanced_summary.json",
            "cv": f"{cfg.out_dir}/ulb_gbdt_cv.json",
            "tuning": f"{cfg.out_dir}/ulb_gbdt_tuning.json",
            "shap_global": f"{cfg.out_dir}/ulb_gbdt_shap_global.json",
        },
    }
    (cfg.out_dir / "serving_config_v1.json").write_text(json.dumps(serving_cfg, indent=2))
    print("Wrote", cfg.out_dir / "serving_config_v1.json")


if __name__ == "__main__":
    main()



