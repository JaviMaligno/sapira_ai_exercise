from __future__ import annotations

"""
Chunked data profiler for fraud sources.

Profiles original source files (BAF, IEEE, PaySim, ULB, Sparkov, DGuard MongoDB)
and emits concise per-source reports aligned with the unified schema mapping.

Outputs:
- reports/{source}_profile.md
- reports/{source}_profile.json

Notes:
- Uses chunked reads for large CSVs to keep memory bounded
- Time coverage derived from source-specific time fields
- For IEEE identity, profiles transaction and identity files separately
- For Sparkov, only customers.csv is available unless transactions are generated
"""

import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import schema as unified_schema


REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# Absolute source paths on this machine
DATASETS = {
    "baf": Path("/home/javier/repos/datasets/Bank Account Fraud (BAF) Dataset Suite - NeurIPS 2022"),
    "ieee": Path("/home/javier/repos/datasets/ieee-fraud-detection"),
    "paysim": Path("/home/javier/repos/datasets/PaySim Synthetic Mobile Money Dataset"),
    "ulb": Path("/home/javier/repos/datasets/ULB Credit Card Fraud Dataset (European Cardholders)"),
    "sparkov": Path("/home/javier/repos/Sparkov_Data_Generation/data"),
}


@dataclass
class NumericStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0  # sum of squares of differences from the current mean
    min_val: float = math.inf
    max_val: float = -math.inf

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        # Filter finite values
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return
        n = finite.size
        new_min = float(np.min(finite))
        new_max = float(np.max(finite))
        self.min_val = min(self.min_val, new_min)
        self.max_val = max(self.max_val, new_max)

        # Welford's algorithm for online variance
        delta = float(np.mean(finite)) - self.mean
        total_n = self.count + n
        self.mean += delta * (n / total_n)
        self.m2 += float(np.var(finite)) * n + (delta**2) * self.count * n / total_n
        self.count = total_n

    @property
    def variance(self) -> Optional[float]:
        if self.count <= 1:
            return None
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> Optional[float]:
        v = self.variance
        return math.sqrt(v) if v is not None and v >= 0 else None


@dataclass
class ColumnProfile:
    non_null_count: int = 0
    null_count: int = 0
    inferred_dtypes: Counter = field(default_factory=Counter)
    numeric_stats: Optional[NumericStats] = None
    top_values: Counter = field(default_factory=Counter)

    def register_series(self, s: pd.Series, topk_sample_limit: int = 200_000, track_numeric: bool = True) -> None:
        non_null = s.notna()
        nn = int(non_null.sum())
        na = int((~non_null).sum())
        self.non_null_count += nn
        self.null_count += na

        # dtype inference bucketized
        dtype_name = str(s.dtype)
        self.inferred_dtypes[dtype_name] += 1

        # Track numeric stats if numeric-like
        if track_numeric and pd.api.types.is_numeric_dtype(s):
            if self.numeric_stats is None:
                self.numeric_stats = NumericStats()
            self.numeric_stats.update(pd.to_numeric(s[non_null].values, errors='coerce'))

        # Top values on a sample to limit memory
        if nn > 0:
            if nn > topk_sample_limit:
                sample = s[non_null].sample(n=topk_sample_limit, random_state=42)
            else:
                sample = s[non_null]
            self.top_values.update(sample.astype(str).head(50_000).tolist())  # cap update volume per chunk

    def to_summary(self, total_rows: int) -> Dict[str, object]:
        fill_rate = (self.non_null_count / total_rows) if total_rows > 0 else 0.0
        numeric = None
        if self.numeric_stats and self.numeric_stats.count > 0:
            numeric = {
                "count": self.numeric_stats.count,
                "mean": self.numeric_stats.mean,
                "std": self.numeric_stats.std,
                "min": self.numeric_stats.min_val if self.numeric_stats.min_val != math.inf else None,
                "max": self.numeric_stats.max_val if self.numeric_stats.max_val != -math.inf else None,
            }
        return {
            "fill_rate": fill_rate,
            "non_null": self.non_null_count,
            "null": self.null_count,
            "dtypes_seen": dict(self.inferred_dtypes),
            "numeric": numeric,
            "top_values": self.top_values.most_common(10),
        }


@dataclass
class SourceProfile:
    source_key: str
    total_rows: int = 0
    columns: Dict[str, ColumnProfile] = field(default_factory=lambda: defaultdict(ColumnProfile))
    notes: List[str] = field(default_factory=list)
    label_counts: Counter = field(default_factory=Counter)
    time_min: Optional[float] = None
    time_max: Optional[float] = None

    def register_chunk(
        self,
        df: pd.DataFrame,
        label_col: Optional[str],
        time_col: Optional[str],
        time_is_numeric_epoch: bool,
        topk_sample_limit: int,
    ) -> None:
        self.total_rows += len(df)
        # Columns
        for col in df.columns:
            self.columns[col].register_series(df[col], topk_sample_limit=topk_sample_limit)

        # Labels
        if label_col and label_col in df.columns:
            vals = df[label_col].dropna()
            # coerce booleans/ints to 0/1 counts
            if pd.api.types.is_bool_dtype(vals):
                counts = vals.astype(int).value_counts()
            else:
                counts = pd.Series(pd.to_numeric(vals, errors='coerce')).dropna().astype(int).value_counts()
            for k, v in counts.items():
                key = int(k)
                self.label_counts[key] += int(v)

        # Time coverage
        if time_col and time_col in df.columns:
            series = df[time_col].dropna()
            if not series.empty:
                if time_is_numeric_epoch:
                    numeric = pd.to_numeric(series, errors='coerce').dropna()
                    if not numeric.empty:
                        mn = float(numeric.min())
                        mx = float(numeric.max())
                        self.time_min = mn if self.time_min is None else min(self.time_min, mn)
                        self.time_max = mx if self.time_max is None else max(self.time_max, mx)
                else:
                    # parse datetime to epoch seconds
                    parsed = pd.to_datetime(series, errors='coerce', utc=True)
                    parsed = parsed.dropna()
                    if not parsed.empty:
                        mn = parsed.min().timestamp()
                        mx = parsed.max().timestamp()
                        self.time_min = mn if self.time_min is None else min(self.time_min, mn)
                        self.time_max = mx if self.time_max is None else max(self.time_max, mx)

    def render_md(self) -> str:
        lines: List[str] = []
        lines.append(f"### Source: {self.source_key}")
        lines.append("")
        lines.append(f"- Total rows scanned: {self.total_rows}")
        if self.label_counts:
            total_labels = sum(self.label_counts.values())
            fraud = self.label_counts.get(1, 0)
            legit = self.label_counts.get(0, 0)
            rate = (fraud / total_labels) if total_labels > 0 else 0.0
            lines.append(f"- Label distribution: 0={legit}, 1={fraud} (fraud rate ~ {rate:.4%})")
        if self.time_min is not None and self.time_max is not None:
            tmin = datetime.fromtimestamp(self.time_min, tz=timezone.utc).isoformat()
            tmax = datetime.fromtimestamp(self.time_max, tz=timezone.utc).isoformat()
            lines.append(f"- Time coverage (approx): {tmin} → {tmax} UTC")
        if self.notes:
            lines.append("- Notes:")
            for n in self.notes:
                lines.append(f"  - {n}")
        lines.append("")
        lines.append("- Top columns by missingness (10):")
        miss = []
        for col, prof in self.columns.items():
            fill = prof.non_null_count / self.total_rows if self.total_rows else 0.0
            miss.append((1 - fill, col))
        miss.sort(reverse=True)
        for frac, col in miss[:10]:
            lines.append(f"  - {col}: missing ~ {frac:.2%}")

        lines.append("")
        lines.append("- Selected column summaries:")
        key_cols = [
            'amount', 'transaction_type', 'merchant_name', 'merchant_lat', 'merchant_long',
            'device_type', 'device_info', 'email_domain_payer', 'email_domain_receiver',
            'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2',
            'dist1', 'dist2', 'risk_score', 'risk_flag', 'event_time_ts', 'event_time_ts_raw', 'event_time_str'
        ]
        for col in key_cols:
            if col in self.columns:
                s = self.columns[col].to_summary(self.total_rows)
                lines.append(f"  - {col}: fill={s['fill_rate']:.2%}, dtypes={list(s['dtypes_seen'].keys())[:2]}")
                if s["numeric"]:
                    num = s["numeric"]
                    mean_str = f"{num['mean']:.3f}" if num.get('mean') is not None else "None"
                    std_val = num.get('std')
                    std_str = f"{std_val:.3f}" if std_val is not None else "None"
                    lines.append(
                        f"    - stats: min={num['min']}, p50≈N/A, max={num['max']}, mean={mean_str}, std={std_str}"
                    )
                if s["top_values"]:
                    top = ", ".join([f"{v} ({c})" for v, c in s["top_values"][:5]])
                    lines.append(f"    - top: {top}")
        return "\n".join(lines)

    def to_json(self) -> Dict[str, object]:
        return {
            "source": self.source_key,
            "total_rows": self.total_rows,
            "labels": dict(self.label_counts),
            "time_min": self.time_min,
            "time_max": self.time_max,
            "columns": {k: v.to_summary(self.total_rows) for k, v in self.columns.items()},
            "notes": self.notes,
        }


def _read_csv_in_chunks(path: Path, sep: str = ',', usecols: Optional[List[str]] = None, chunksize: int = 200_000) -> Iterable[pd.DataFrame]:
    return pd.read_csv(path, sep=sep, chunksize=chunksize, low_memory=False, usecols=usecols)


def profile_baf(row_limit: int = 1_000_000, topk_sample_limit: int = 200_000) -> SourceProfile:
    prof = SourceProfile(source_key="baf")
    # Files: Base.csv, Variant I..V
    files = [
        DATASETS["baf"] / "Base.csv",
        DATASETS["baf"] / "Variant I.csv",
        DATASETS["baf"] / "Variant II.csv",
        DATASETS["baf"] / "Variant III.csv",
        DATASETS["baf"] / "Variant IV.csv",
        DATASETS["baf"] / "Variant V.csv",
    ]
    # Columns of interest
    ren = unified_schema.BAF_RENAME
    as_is = unified_schema.BAF_AS_IS
    key_cols = list(ren.keys()) + list(as_is)
    scanned = 0
    for f in files:
        if not f.exists():
            prof.notes.append(f"Missing file: {f}")
            continue
        for chunk in _read_csv_in_chunks(f, usecols=[c for c in key_cols if c != "x1" and c != "x2"]):
            # Map label col for counts
            label_col = "fraud_bool"
            # No canonical timestamp; capture month/days_since_request coverage as notes
            prof.register_chunk(chunk, label_col=label_col, time_col=None, time_is_numeric_epoch=False, topk_sample_limit=topk_sample_limit)
            scanned += len(chunk)
            if scanned >= row_limit:
                break
        if scanned >= row_limit:
            break
    if scanned == 0:
        prof.notes.append("No rows scanned (files missing?)")
    return prof


def profile_ieee(row_limit: int = 1_000_000, topk_sample_limit: int = 200_000) -> List[SourceProfile]:
    # Profile transactions and identity separately to avoid heavy join here
    tx_prof = SourceProfile(source_key="ieee_transaction")
    id_prof = SourceProfile(source_key="ieee_identity")

    tx_path = DATASETS["ieee"] / "train_transaction.csv"
    id_path = DATASETS["ieee"] / "train_identity.csv"

    # Transaction columns
    tx_cols = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2", "dist1", "dist2", "P_emaildomain", "R_emaildomain"]
    scanned = 0
    if tx_path.exists():
        for chunk in _read_csv_in_chunks(tx_path, usecols=tx_cols):
            tx_prof.register_chunk(
                chunk.rename(columns=unified_schema.IEEE_RENAME),
                label_col="is_fraud",
                time_col="event_time_ts_raw",
                time_is_numeric_epoch=True,
                topk_sample_limit=topk_sample_limit,
            )
            scanned += len(chunk)
            if scanned >= row_limit:
                break
    else:
        tx_prof.notes.append(f"Missing file: {tx_path}")

    # Identity columns
    id_cols = ["TransactionID", "DeviceType", "DeviceInfo"] + [f"id_{i:02d}" for i in range(1, 39)]
    scanned = 0
    if id_path.exists():
        for chunk in _read_csv_in_chunks(id_path, usecols=id_cols):
            id_prof.register_chunk(
                chunk.rename(columns=unified_schema.IEEE_RENAME),
                label_col=None,
                time_col=None,
                time_is_numeric_epoch=False,
                topk_sample_limit=topk_sample_limit,
            )
            scanned += len(chunk)
            if scanned >= row_limit:
                break
    else:
        id_prof.notes.append(f"Missing file: {id_path}")

    return [tx_prof, id_prof]


def profile_paysim(row_limit: int = 2_000_000, topk_sample_limit: int = 200_000) -> SourceProfile:
    prof = SourceProfile(source_key="paysim")
    path = DATASETS["paysim"] / "PS_20174392719_1491204439457_log.csv"
    usecols = [
        "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
        "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
    ]
    scanned = 0
    if path.exists():
        for chunk in _read_csv_in_chunks(path, usecols=usecols):
            prof.register_chunk(
                chunk.rename(columns=unified_schema.PAYSIM_RENAME),
                label_col="is_fraud",
                time_col="event_time_ts",  # step used as pseudo time; epoch mapping omitted here
                time_is_numeric_epoch=True,
                topk_sample_limit=topk_sample_limit,
            )
            scanned += len(chunk)
            if scanned >= row_limit:
                break
    else:
        prof.notes.append(f"Missing file: {path}")
    return prof


def profile_ulb(row_limit: int = 2_000_000, topk_sample_limit: int = 200_000) -> List[SourceProfile]:
    def profile_one(path: Path, tag: str) -> SourceProfile:
        p = SourceProfile(source_key=f"ulb_{tag}")
        usecols = [
            "trans_num", "unix_time", "category", "amt", "merchant", "merch_lat", "merch_long", "is_fraud",
            "first", "last", "gender", "street", "city", "state", "zip", "lat", "long", "city_pop", "job", "dob",
        ]
        if path.exists():
            scanned = 0
            for chunk in _read_csv_in_chunks(path, usecols=usecols):
                p.register_chunk(
                    chunk.rename(columns=unified_schema.ULB_RENAME),
                    label_col="is_fraud",
                    time_col="event_time_ts",
                    time_is_numeric_epoch=True,
                    topk_sample_limit=topk_sample_limit,
                )
                scanned += len(chunk)
                if scanned >= row_limit:
                    break
        else:
            p.notes.append(f"Missing file: {path}")
        return p

    train = profile_one(DATASETS["ulb"] / "fraudTrain.csv", tag="train")
    test = profile_one(DATASETS["ulb"] / "fraudTest.csv", tag="test")
    return [train, test]


def profile_sparkov(row_limit: int = 2_000_000, topk_sample_limit: int = 200_000) -> List[SourceProfile]:
    profiles: List[SourceProfile] = []

    # Customers file
    cust_prof = SourceProfile(source_key="sparkov_customers")
    cust_path = DATASETS["sparkov"] / "customers.csv"
    if cust_path.exists():
        usecols = [
            "ssn", "cc_num", "first", "last", "gender", "street", "city", "state", "zip",
            "lat", "long", "city_pop", "job", "dob", "acct_num", "profile"
        ]
        scanned = 0
        for chunk in _read_csv_in_chunks(cust_path, sep='|', usecols=usecols):
            cust_prof.register_chunk(
                chunk,
                label_col=None,
                time_col=None,
                time_is_numeric_epoch=False,
                topk_sample_limit=topk_sample_limit,
            )
            scanned += len(chunk)
            if scanned >= row_limit:
                break
    else:
        cust_prof.notes.append(f"Missing file: {cust_path}")
    profiles.append(cust_prof)

    # Generated transactions may not exist yet (pipe-delimited). If present, profile.
    tx_candidates = list(Path(DATASETS["sparkov"]).glob("*.psv")) + list(Path(DATASETS["sparkov"]).glob("*.csv|*.psv"))
    for tx_path in tx_candidates:
        if not tx_path.exists():
            continue
        if tx_path.name == "customers.csv":
            continue
        tx_prof = SourceProfile(source_key=f"sparkov_tx:{tx_path.name}")
        try:
            # Try pipe-delimited first, fallback to comma
            for chunk in _read_csv_in_chunks(tx_path, sep='|'):
                tx_prof.register_chunk(
                    chunk.rename(columns=unified_schema.SPARKOV_RENAME),
                    label_col="is_fraud" if "is_fraud" in chunk.columns else None,
                    time_col="event_time_ts" if "unix_time" in chunk.columns or "event_time_ts" in chunk.columns else None,
                    time_is_numeric_epoch=True,
                    topk_sample_limit=topk_sample_limit,
                )
                if tx_prof.total_rows >= row_limit:
                    break
        except Exception:
            for chunk in _read_csv_in_chunks(tx_path, sep=','):
                tx_prof.register_chunk(
                    chunk.rename(columns=unified_schema.SPARKOV_RENAME),
                    label_col="is_fraud" if "is_fraud" in chunk.columns else None,
                    time_col="event_time_ts" if "unix_time" in chunk.columns or "event_time_ts" in chunk.columns else None,
                    time_is_numeric_epoch=True,
                    topk_sample_limit=topk_sample_limit,
                )
                if tx_prof.total_rows >= row_limit:
                    break
        profiles.append(tx_prof)

    if len(profiles) == 1:
        cust_prof.notes.append("No Sparkov transaction files found. Generate transactions before modeling.")
    return profiles


def profile_dguard_mongo(limit_docs: int = 50_000) -> Optional[SourceProfile]:
    try:
        from pymongo import MongoClient
    except Exception:
        return None

    from .settings import MONGO_URI
    client = None
    prof = SourceProfile(source_key="dguard_tx")
    try:
        client = MongoClient(
            MONGO_URI,
            uuidRepresentation='standard',
            tz_aware=True,
            serverSelectionTimeoutMS=8000,
            authSource='admin',
        )
        coll = client["dguard"]["bank_transactions"]
        fields = {
            'uuid': 1, 'user_id': 1, 'account_id': 1, 'operation_date': 1, 'amount': 1, 'currency': 1,
            'description': 1, 'operation_type': 1, 'fraud_score': 1, 'is_suspicious': 1, 'fraud_status': 1,
            'merchant_clean_name': 1, 'categories': 1, 'balance': 1,
        }
        cursor = coll.find({}, fields).limit(limit_docs)
        batch: List[dict] = []
        batch_size = 5000
        for doc in cursor:
            batch.append(doc)
            if len(batch) >= batch_size:
                df = pd.DataFrame(batch)
                prof.register_chunk(
                    df.rename(columns=unified_schema.DGUARD_RENAME),
                    label_col=None,
                    time_col="event_time_str",
                    time_is_numeric_epoch=False,
                    topk_sample_limit=100_000,
                )
                batch.clear()
        if batch:
            df = pd.DataFrame(batch)
            prof.register_chunk(
                df.rename(columns=unified_schema.DGUARD_RENAME),
                label_col=None,
                time_col="event_time_str",
                time_is_numeric_epoch=False,
                topk_sample_limit=100_000,
            )
        if prof.total_rows == 0:
            prof.notes.append("No documents found in bank_transactions or connection blocked.")
    except Exception as e:
        if prof.notes is None:
            prof.notes = []
        prof.notes.append(f"Mongo profiling failed: {e}")
    finally:
        try:
            if client:
                client.close()
        except Exception:
            pass
    return prof


def write_report(prof: SourceProfile) -> None:
    # JSON
    out_json = REPORTS_DIR / f"{prof.source_key}_profile.json"
    with out_json.open("w") as f:
        json.dump(prof.to_json(), f, indent=2)
    # Markdown
    out_md = REPORTS_DIR / f"{prof.source_key}_profile.md"
    out_md.write_text(prof.render_md())


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Profile fraud data sources")
    parser.add_argument("--sources", nargs="*", default=["baf", "ieee", "paysim", "ulb", "sparkov", "dguard"], help="Which sources to profile")
    parser.add_argument("--row-limit", type=int, default=1_000_000, help="Max rows per file/source to scan")
    parser.add_argument("--topk-sample-limit", type=int, default=200_000, help="Max rows per chunk to sample for top values")
    args = parser.parse_args()

    to_run = set([s.lower() for s in args.sources])

    if "baf" in to_run:
        prof = profile_baf(row_limit=args.row_limit, topk_sample_limit=args.topk_sample_limit)
        write_report(prof)

    if "ieee" in to_run:
        profs = profile_ieee(row_limit=args.row_limit, topk_sample_limit=args.topk_sample_limit)
        for p in profs:
            write_report(p)

    if "paysim" in to_run:
        prof = profile_paysim(row_limit=args.row_limit * 2, topk_sample_limit=args.topk_sample_limit)
        write_report(prof)

    if "ulb" in to_run:
        for p in profile_ulb(row_limit=args.row_limit, topk_sample_limit=args.topk_sample_limit):
            write_report(p)

    if "sparkov" in to_run:
        for p in profile_sparkov(row_limit=args.row_limit, topk_sample_limit=args.topk_sample_limit):
            write_report(p)

    if "dguard" in to_run:
        p = profile_dguard_mongo(limit_docs=50_000)
        if p is not None:
            write_report(p)


if __name__ == "__main__":
    main()


