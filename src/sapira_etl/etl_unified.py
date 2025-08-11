from __future__ import annotations

import argparse
from pathlib import Path
import os
import pandas as pd
from typing import Optional

from .utils import load_env, get_db_config, get_hash_salt, stable_hash
from .writers import write_parquet, write_postgres, write_parquet_parts

try:
    from pymongo import MongoClient  # type: ignore
except Exception:  # pragma: no cover
    MongoClient = None  # type: ignore


def _hash_series(s: pd.Series, salt: str) -> pd.Series:
    return s.astype(str).map(lambda v: stable_hash(v if v != 'nan' else None, salt))


def load_ieee_train(base_dir: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    tx = pd.read_csv(base_dir / 'train_transaction.csv', nrows=nrows)
    idn = pd.read_csv(base_dir / 'train_identity.csv', nrows=nrows)
    df = tx.merge(idn, on='TransactionID', how='left')
    df.rename(columns={
        'TransactionID': 'transaction_id',
        'isFraud': 'is_fraud',
        'TransactionAmt': 'amount',
        'ProductCD': 'product_code',
        'P_emaildomain': 'email_domain_payer',
        'R_emaildomain': 'email_domain_receiver',
        'DeviceType': 'device_type',
        'DeviceInfo': 'device_info',
    }, inplace=True)
    df['dataset'] = 'ieee_train'
    df['event_time_ts_raw'] = df['TransactionDT']
    return df


def load_ulb(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=nrows)
    df.rename(columns={
        'trans_num': 'transaction_id',
        'unix_time': 'event_time_ts',
        'category': 'transaction_type',
        'amt': 'amount',
        'merchant': 'merchant_name',
        'merch_lat': 'merchant_lat',
        'merch_long': 'merchant_long',
        'is_fraud': 'is_fraud',
    }, inplace=True)
    return df


def load_baf_all(base_dir: Path, nrows: Optional[int] = None) -> list[pd.DataFrame]:
    mapping = {
        'fraud_bool': 'is_fraud',
        'payment_type': 'transaction_type',
        'session_length_in_minutes': 'session_length_minutes',
    }
    files = [
        ('Base.csv', 'baf_base'),
        ('Variant I.csv', 'baf_var_i'),
        ('Variant II.csv', 'baf_var_ii'),
        ('Variant III.csv', 'baf_var_iii'),
        ('Variant IV.csv', 'baf_var_iv'),
        ('Variant V.csv', 'baf_var_v'),
    ]
    frames: list[pd.DataFrame] = []
    for fname, ds in files:
        fpath = base_dir / fname
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath, nrows=nrows)
        df.rename(columns=mapping, inplace=True)
        df['dataset'] = ds
        frames.append(df)
    return frames


def load_paysim(path: Path, nrows: Optional[int] = None, epoch0: int = 1546300800) -> pd.DataFrame:
    # epoch0 default is 2019-01-01T00:00:00Z
    df = pd.read_csv(path, nrows=nrows)
    df.rename(columns={
        'type': 'transaction_type',
        'amount': 'amount',
        'isFraud': 'is_fraud',
        'nameDest': 'counterparty_id',
        'oldbalanceOrg': 'orig_balance_before',
        'newbalanceOrg': 'orig_balance_after',
        'oldbalanceDest': 'dest_balance_before',
        'newbalanceDest': 'dest_balance_after',
    }, inplace=True)
    # derive event_time_ts from step (hours)
    if 'step' in df.columns:
        df['event_time_ts'] = epoch0 + df['step'].astype(int) * 3600
    return df


def detect_pipe_delimited_files(base_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    for root in [base_dir, base_dir / 'data']:
        if not root.exists():
            continue
        for p in root.glob('**/*.csv'):
            try:
                with p.open('r') as f:
                    head = f.readline()
                    if '|' in head and ('trans_num' in head or 'unix_time' in head) and ('cc_num' in head or 'first|' in head):
                        candidates.append(p)
            except Exception:
                continue
    return candidates


def load_sparkov_transactions(base_dir: Path, nrows: Optional[int] = None) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for fpath in detect_pipe_delimited_files(base_dir):
        try:
            df = pd.read_csv(fpath, sep='|', nrows=nrows)
        except Exception:
            continue
        # Normalize columns where present
        rename_map = {
            'trans_num': 'transaction_id',
            'unix_time': 'event_time_ts',
            'category': 'transaction_type',
            'amt': 'amount',
            'merchant': 'merchant_name',
            'merch_lat': 'merchant_lat',
            'merch_long': 'merchant_long',
            'is_fraud': 'is_fraud',
        }
        for k, v in rename_map.items():
            if k in df.columns:
                df.rename(columns={k: v}, inplace=True)
        df['dataset'] = 'sparkov_tx'
        frames.append(df)
    return frames


def load_dguard_bank_transactions(mongo_uri: str, limit: Optional[int] = None) -> Optional[pd.DataFrame]:
    if MongoClient is None:
        print('pymongo not available; skipping Mongo load')
        return None
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        dbname = (mongo_uri.rsplit('/', 1)[-1] or 'dguard').split('?')[0]
        db = client[dbname]
        coll = db['bank_transactions']
        proj = {
            'uuid': 1,
            'user_id': 1,
            'account_id': 1,
            'operation_date': 1,
            'amount': 1,
            'currency': 1,
            'description': 1,
            'operation_type': 1,
            'fraud_score': 1,
            'is_suspicious': 1,
            'fraud_status': 1,
            'merchant_clean_name': 1,
            'categories': 1,
            'balance': 1,
        }
        cursor = coll.find({}, proj)
        if limit:
            cursor = cursor.limit(int(limit))
        docs = list(cursor)
        if not docs:
            return None
        df = pd.DataFrame(docs)
        df.rename(columns={
            'uuid': 'transaction_id',
            'merchant_clean_name': 'merchant_name',
            'operation_type': 'transaction_type',
            'fraud_score': 'risk_score',
            'is_suspicious': 'risk_flag',
        }, inplace=True)
        # event_time_str + attempt to parse to ts
        if 'operation_date' in df.columns:
            df['event_time_str'] = df['operation_date']
            # leave parsing for downstream; timestamps may vary in format
        df['dataset'] = 'dguard_tx'
        return df
    except Exception as e:
        print(f"Mongo load failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Unified ETL to Parquet and Postgres')
    parser.add_argument('--env', type=str, default=None, help='Path to env file')
    parser.add_argument('--parquet-out', type=str, default='sapira_ai_exercise/data/unified', help='Output dir for Parquet')
    parser.add_argument('--write-postgres', action='store_true', help='Write to Postgres as well')
    parser.add_argument('--pg-table', type=str, default='unified_transactions', help='Destination table name')
    parser.add_argument('--limit-rows', type=int, default=None, help='Optional row limit per source for testing')
    parser.add_argument('--include-sparkov', action='store_true', help='Include Sparkov generated transactions if present')
    parser.add_argument('--include-mongo', action='store_true', help='Include DGuard Mongo bank_transactions (requires MONGO_URI)')
    parser.add_argument('--chunk-rows', type=int, default=100_000, help='Process and flush in chunks of this many rows (0 disables chunking)')
    args = parser.parse_args()

    load_env(args.env)
    salt = get_hash_salt()

    # Streaming write state
    part_counters = None
    wrote_any = False

    def process_df(df_in: pd.DataFrame) -> None:
        nonlocal part_counters, wrote_any
        if df_in is None or df_in.empty:
            return
        wrote_any = True
        out_dir = Path(args.parquet_out)
        if args.chunk_rows and args.chunk_rows > 0:
            total_rows_local = len(df_in)
            start_local = 0
            while start_local < total_rows_local:
                end_local = min(start_local + args.chunk_rows, total_rows_local)
                chunk_local = df_in.iloc[start_local:end_local]
                part_counters = write_parquet_parts(chunk_local, out_dir, part_counters)
                if args.write_postgres:
                    db = get_db_config()
                    write_postgres(chunk_local, args.pg_table, db)
                start_local = end_local
        else:
            # Single-shot for this source
            part_counters = write_parquet_parts(df_in, out_dir, part_counters)
            if args.write_postgres:
                db = get_db_config()
                write_postgres(df_in, args.pg_table, db)

    # IEEE train (optionally limited)
    ieee_dir = Path('/home/javier/repos/datasets/ieee-fraud-detection')
    if ieee_dir.exists():
        df_ieee = load_ieee_train(ieee_dir, nrows=args.limit_rows)
        # Example hash: card1 as token
        if 'card1' in df_ieee.columns:
            df_ieee['card_token_hash'] = _hash_series(df_ieee['card1'], salt)
        process_df(df_ieee)

    # ULB train/test
    ulb_train = Path('/home/javier/repos/datasets/ULB Credit Card Fraud Dataset (European Cardholders)/fraudTrain.csv')
    ulb_test = Path('/home/javier/repos/datasets/ULB Credit Card Fraud Dataset (European Cardholders)/fraudTest.csv')
    if ulb_train.exists():
        df_ut = load_ulb(ulb_train, nrows=args.limit_rows)
        df_ut['dataset'] = 'ulb_train'
        process_df(df_ut)
    if ulb_test.exists():
        df_uv = load_ulb(ulb_test, nrows=args.limit_rows)
        df_uv['dataset'] = 'ulb_test'
        process_df(df_uv)

    # BAF base and variants
    baf_dir = Path('/home/javier/repos/datasets/Bank Account Fraud (BAF) Dataset Suite - NeurIPS 2022')
    if baf_dir.exists():
        for df_b in load_baf_all(baf_dir, nrows=args.limit_rows):
            process_df(df_b)

    # PaySim
    paysim_path = Path('/home/javier/repos/datasets/PaySim Synthetic Mobile Money Dataset/PS_20174392719_1491204439457_log.csv')
    if paysim_path.exists():
        df_ps = load_paysim(paysim_path, nrows=args.limit_rows)
        df_ps['dataset'] = 'paysim'
        process_df(df_ps)

    # Sparkov (optional discovery of pipe-delimited transactions)
    if args.include_sparkov:
        sparkov_dir = Path('/home/javier/repos/Sparkov_Data_Generation')
        if sparkov_dir.exists():
            for df_sp in load_sparkov_transactions(sparkov_dir, nrows=args.limit_rows):
                process_df(df_sp)

    # DGuard Mongo (optional)
    if args.include_mongo:
        mongo_uri = os.getenv('MONGO_URI', '')
        if mongo_uri:
            df_m = load_dguard_bank_transactions(mongo_uri, limit=args.limit_rows)
            if df_m is not None:
                process_df(df_m)

    if not wrote_any:
        print('No sources found. Exiting.')
        return

    print('ETL completed.')


if __name__ == '__main__':
    main()


