from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer

from fraud_mvp.gbdt_ulb_enhanced import engineer_enhanced


def harmonize_input(df: pd.DataFrame) -> pd.DataFrame:
    cols = set(df.columns)
    # ULB raw
    if {"trans_num", "unix_time", "category", "amt", "merchant"}.issubset(cols):
        df = df.rename(
            columns={
                "trans_num": "transaction_id",
                "unix_time": "event_time_ts",
                "category": "operation_type",
                "amt": "amount",
                "merchant": "merchant_name",
                "is_fraud": "is_fraud",
            }
        )
        df["event_time"] = pd.to_datetime(df["event_time_ts"], unit="s", utc=True)
        df["dataset"] = df.get("dataset", "ULB")
        df["cc_num"] = df.get("cc_num", -1)
        return df
    # IEEE raw
    if {"TransactionID", "TransactionDT", "TransactionAmt", "ProductCD"}.issubset(cols):
        df = df.rename(
            columns={
                "TransactionID": "transaction_id",
                "TransactionDT": "event_time_ts",
                "TransactionAmt": "amount",
                "ProductCD": "operation_type",
                "isFraud": "is_fraud",
            }
        )
        base = pd.Timestamp("2017-12-01", tz="UTC")
        df["event_time"] = base + pd.to_timedelta(df["event_time_ts"], unit="s")
        merch = df.get("P_emaildomain", pd.Series([np.nan] * len(df))).fillna(df.get("card4", "")).astype(str)
        df["merchant_name"] = merch.fillna("UNK").astype(str)
        df["cc_num"] = df.get("card1", -1).fillna(-1).astype(int)
        df["dataset"] = df.get("dataset", "IEEE")
        return df
    # Assume already harmonized
    return df


def compute_scores_with_calibration(pipeline, isotonic, X_df: pd.DataFrame) -> np.ndarray:
    if hasattr(pipeline.named_steps["gbdt"], "predict_proba"):
        raw = pipeline.named_steps["gbdt"].predict_proba(pipeline.named_steps["prep"].transform(X_df))[:, 1]
    else:
        raw = pipeline.named_steps["gbdt"].predict(X_df)
    return isotonic.transform(raw)


def load_thresholds(artifacts_dir: Path, dataset: str, alert_frac: float) -> Dict[str, float]:
    key = str(alert_frac).replace(".", "p")
    if dataset.upper() == "IEEE":
        path = artifacts_dir / f"gbdt_thresholds_IEEE_{key}.json"
    else:
        path = artifacts_dir / f"gbdt_per_category_thresholds_{key}.json"
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--artifacts_dir", default="reports/phase2/ulb_gbdt")
    parser.add_argument("--alert_frac", type=float, default=0.005)
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)

    # Load artifacts
    pipeline = joblib.load(artifacts_dir / "pipeline.pkl")
    isotonic = joblib.load(artifacts_dir / "isotonic.pkl")
    if_pipe = joblib.load(artifacts_dir / "if_pipe.pkl")
    freq_map = json.loads((artifacts_dir / "merchant_freq_map.json").read_text())
    rule_params = json.loads((artifacts_dir / "rule_params.json").read_text()) if (artifacts_dir / "rule_params.json").exists() else {"p995": None}
    p995 = rule_params.get("p995")

    # Load data
    df_raw = pd.read_csv(args.input_csv)
    df_h = harmonize_input(df_raw)
    dataset = str(df_h.get("dataset", "ULB").iloc[0]).upper()

    # Engineer features
    df_feat, num_cols, cat_cols = engineer_enhanced(df_h)
    # Map merchant frequency
    df_feat["merchant_freq_global"] = df_feat["merchant_name"].fillna("UNK").map(freq_map).fillna(0.0)
    # IF anomaly score
    if_scores = -if_pipe.named_steps["if"].decision_function(if_pipe.named_steps["prep"].transform(df_feat))
    df_feat["if_anomaly_score"] = if_scores
    # Rule score
    if p995 is None or (isinstance(p995, float) and np.isnan(p995)):
        p995 = float(np.nanpercentile(df_feat["abs_amount"], 99.5)) if df_feat["abs_amount"].notna().any() else np.nan
    high_amt = ((df_feat["abs_amount"] >= p995) & df_feat["abs_amount"].notna()).astype(int)
    new_merch_high = ((df_feat["is_new_merchant_for_card"] == 1) & (df_feat["abs_amount"] >= p995)).astype(int)
    rapid = (df_feat["txn_count_24h"] >= 10).astype(int)
    df_feat["rule_score"] = 1.5 * high_amt + 1.0 * new_merch_high + 0.5 * rapid

    # Score with calibration
    scores = compute_scores_with_calibration(pipeline, isotonic, df_feat)

    # Thresholds
    thresholds = load_thresholds(artifacts_dir, dataset, args.alert_frac)
    cats = df_feat["operation_type"].astype(str).values
    flags = np.array([scores[i] >= thresholds.get(str(cats[i]), np.inf) for i in range(len(scores))])

    # SHAP explanations for flagged
    pre: ColumnTransformer = pipeline.named_steps["prep"]
    X_mat = pre.transform(df_feat)
    explainer = shap.Explainer(pipeline.named_steps["gbdt"])
    flagged_idx = np.where(flags)[0]
    shap_vals_flagged = explainer(X_mat[flagged_idx])

    alerts: List[Dict] = []
    for rank, idx in enumerate(flagged_idx):
        vals = shap_vals_flagged.values[rank]
        contrib_order = np.argsort(np.abs(vals))[::-1][:5]
        raw_names = pre.get_feature_names_out()
        def clean_name(n: str) -> str:
            n = n.replace("num__", "").replace("cat__oh__", "").replace("cat__", "")
            return n
        feat_names = [clean_name(n) for n in raw_names]
        contribs = [{"feature": feat_names[j], "shap_value": float(vals[j])} for j in contrib_order]
        row = df_feat.iloc[idx]
        alerts.append({
            "rank": int(rank + 1),
            "score": float(scores[idx]),
            "transaction_id": str(df_h.iloc[idx].get("transaction_id", "")),
            "event_time": str(row.get("event_time")),
            "operation_type": str(row.get("operation_type")),
            "amount": float(row.get("amount", np.nan)),
            "merchant_name": str(row.get("merchant_name")),
            "dataset": dataset,
            "reasons": contribs,
        })

    # Save outputs
    pd.DataFrame(alerts).to_csv(args.output_csv, index=False)
    Path(args.output_json).write_text(json.dumps({"alert_frac": args.alert_frac, "count": len(alerts), "alerts": alerts}, indent=2))


if __name__ == "__main__":
    main()




