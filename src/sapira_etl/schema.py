from __future__ import annotations

"""
Unified schema dictionary for the consolidated fraud datasets.

Types are Postgres column types to be used when creating/loading the
`unified_transactions` table (or any derived table).

Note: This schema was inferred from sampling and the normalization logic in
`etl_unified.py`. It favors permissive numeric types and TEXT for
categoricals. Time-like numeric fields are stored as BIGINT epoch seconds
to avoid timezone parsing assumptions.
"""

from typing import Dict, List, Tuple
from pathlib import Path
import csv


# Postgres types per column in the unified dataset
UNIFIED_SCHEMA_PG: Dict[str, str] = {
    # IEEE / identity / transaction engineered features
    'C1': 'DOUBLE PRECISION', 'C2': 'DOUBLE PRECISION', 'C3': 'DOUBLE PRECISION',
    'C4': 'DOUBLE PRECISION', 'C5': 'DOUBLE PRECISION', 'C6': 'DOUBLE PRECISION',
    'C7': 'DOUBLE PRECISION', 'C8': 'DOUBLE PRECISION', 'C9': 'DOUBLE PRECISION',
    'C10': 'DOUBLE PRECISION', 'C11': 'DOUBLE PRECISION', 'C12': 'DOUBLE PRECISION',
    'C13': 'DOUBLE PRECISION', 'C14': 'DOUBLE PRECISION',
    'D1': 'DOUBLE PRECISION', 'D2': 'DOUBLE PRECISION', 'D3': 'DOUBLE PRECISION',
    'D4': 'DOUBLE PRECISION', 'D5': 'DOUBLE PRECISION', 'D6': 'DOUBLE PRECISION',
    'D7': 'DOUBLE PRECISION', 'D8': 'DOUBLE PRECISION', 'D9': 'DOUBLE PRECISION',
    'D10': 'DOUBLE PRECISION', 'D11': 'DOUBLE PRECISION', 'D12': 'DOUBLE PRECISION',
    'D13': 'DOUBLE PRECISION', 'D14': 'DOUBLE PRECISION', 'D15': 'DOUBLE PRECISION',
    'M1': 'TEXT', 'M2': 'TEXT', 'M3': 'TEXT', 'M4': 'TEXT', 'M5': 'TEXT', 'M6': 'TEXT',
    'M7': 'TEXT', 'M8': 'TEXT', 'M9': 'TEXT',
    'TransactionDT': 'DOUBLE PRECISION',
    # IEEE V1..V339
    **{f'V{i}': 'DOUBLE PRECISION' for i in range(1, 340)},
    # Common normalized fields across sources
    'transaction_id': 'TEXT',
    'is_fraud': 'BOOLEAN',
    'amount': 'DOUBLE PRECISION',
    'transaction_type': 'TEXT',
    'merchant_name': 'TEXT',
    'merchant_lat': 'DOUBLE PRECISION',
    'merchant_long': 'DOUBLE PRECISION',
    'dataset': 'TEXT',
    # Time representations
    'event_time_ts': 'BIGINT',
    'event_time_ts_raw': 'BIGINT',
    # Identity / card features from IEEE
    'product_code': 'TEXT', 'card1': 'DOUBLE PRECISION', 'card2': 'DOUBLE PRECISION',
    'card3': 'DOUBLE PRECISION', 'card4': 'TEXT', 'card5': 'DOUBLE PRECISION',
    'card6': 'TEXT', 'addr1': 'DOUBLE PRECISION', 'addr2': 'DOUBLE PRECISION',
    'dist1': 'DOUBLE PRECISION', 'dist2': 'DOUBLE PRECISION',
    'email_domain_payer': 'TEXT', 'email_domain_receiver': 'TEXT',
    'device_type': 'TEXT', 'device_info': 'TEXT',
    # ULB
    'merchant': 'TEXT',  # retained for some raw files if present
    # Sparkov / ULB geo and person
    'cc_num': 'DOUBLE PRECISION', 'first': 'TEXT', 'last': 'TEXT', 'gender': 'TEXT',
    'street': 'TEXT', 'city': 'TEXT', 'state': 'TEXT', 'zip': 'DOUBLE PRECISION',
    'lat': 'DOUBLE PRECISION', 'long': 'DOUBLE PRECISION', 'city_pop': 'DOUBLE PRECISION',
    'job': 'TEXT', 'dob': 'TEXT',
    # Sparkov derived
    'trans_date_trans_time': 'TEXT', 'trans_date': 'TEXT', 'trans_time': 'TEXT',
    'month': 'DOUBLE PRECISION', 'x1': 'DOUBLE PRECISION', 'x2': 'DOUBLE PRECISION',
    # PaySim mapping
    'counterparty_id': 'TEXT', 'orig_balance_before': 'DOUBLE PRECISION',
    'newbalanceOrig': 'DOUBLE PRECISION', 'dest_balance_before': 'DOUBLE PRECISION',
    'dest_balance_after': 'DOUBLE PRECISION', 'nameOrig': 'TEXT', 'step': 'DOUBLE PRECISION',
    'isFlaggedFraud': 'DOUBLE PRECISION',
    # BAF demographic/behavioral
    'email_is_free': 'DOUBLE PRECISION', 'employment_status': 'TEXT',
    'housing_status': 'TEXT', 'phone_home_valid': 'DOUBLE PRECISION',
    'phone_mobile_valid': 'DOUBLE PRECISION', 'bank_months_count': 'DOUBLE PRECISION',
    'has_other_cards': 'DOUBLE PRECISION', 'proposed_credit_limit': 'DOUBLE PRECISION',
    'foreign_request': 'DOUBLE PRECISION', 'source': 'TEXT', 'session_length_minutes': 'DOUBLE PRECISION',
    'device_os': 'TEXT', 'keep_alive_session': 'DOUBLE PRECISION',
    'device_distinct_emails_8w': 'DOUBLE PRECISION', 'device_fraud_count': 'DOUBLE PRECISION',
    'zip_count_4w': 'DOUBLE PRECISION', 'velocity_6h': 'DOUBLE PRECISION',
    'velocity_24h': 'DOUBLE PRECISION', 'velocity_4w': 'DOUBLE PRECISION',
    'bank_branch_count_8w': 'DOUBLE PRECISION', 'date_of_birth_distinct_emails_4w': 'DOUBLE PRECISION',
    'credit_risk_score': 'DOUBLE PRECISION', 'customer_age': 'DOUBLE PRECISION',
    'days_since_request': 'DOUBLE PRECISION', 'intended_balcon_amount': 'DOUBLE PRECISION',
    'current_address_months_count': 'DOUBLE PRECISION', 'prev_address_months_count': 'DOUBLE PRECISION',
    'name_email_similarity': 'DOUBLE PRECISION', 'income': 'DOUBLE PRECISION',
    # Misc IDs that may appear in some sources
    'acct_num': 'DOUBLE PRECISION', 'profile': 'TEXT', 'ssn': 'TEXT', 'Unnamed: 0': 'DOUBLE PRECISION',
    # Optional risk fields from Mongo
    'risk_score': 'DOUBLE PRECISION', 'risk_flag': 'BOOLEAN', 'event_time_str': 'TEXT',
}


# Core columns frequently used for modeling or analytics
CORE_COLUMNS = [
    'transaction_id', 'dataset', 'event_time_ts', 'amount', 'transaction_type',
    'is_fraud', 'merchant_name', 'merchant_lat', 'merchant_long',
]


def generate_create_table_sql(schema_name: str, table: str, schema: Dict[str, str] | None = None) -> str:
    """Generate a CREATE TABLE statement for Postgres from the schema dictionary.

    All columns are created as nullable; adjust downstream if needed.
    """
    columns = schema or UNIFIED_SCHEMA_PG
    cols_sql = ",\n  ".join([f'"{c}" {t}' for c, t in columns.items()])
    return f"CREATE TABLE IF NOT EXISTS {schema_name}.\"{table}\" (\n  {cols_sql}\n);"


# Source-to-unified mappings, based on transformations in etl_unified.py
# Each entry maps original column name -> unified column name
IEEE_RENAME = {
    'TransactionID': 'transaction_id',
    'isFraud': 'is_fraud',
    'TransactionAmt': 'amount',
    'ProductCD': 'product_code',
    'P_emaildomain': 'email_domain_payer',
    'R_emaildomain': 'email_domain_receiver',
    'DeviceType': 'device_type',
    'DeviceInfo': 'device_info',
    # Derived
    'TransactionDT': 'event_time_ts_raw',
}
IEEE_AS_IS: List[str] = (
    [f'C{i}' for i in range(1, 15)]
    + [f'D{i}' for i in range(1, 16)]
    + [f'M{i}' for i in range(1, 10)]
    + [f'V{i}' for i in range(1, 340)]
    + ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1', 'dist2']
)

ULB_RENAME = {
    'trans_num': 'transaction_id',
    'unix_time': 'event_time_ts',
    'category': 'transaction_type',
    'amt': 'amount',
    'merchant': 'merchant_name',
    'merch_lat': 'merchant_lat',
    'merch_long': 'merchant_long',
    'is_fraud': 'is_fraud',
}
ULB_AS_IS = ['first', 'last', 'gender', 'street', 'city', 'state', 'zip', 'lat', 'long', 'city_pop', 'job', 'dob']

BAF_RENAME = {
    'fraud_bool': 'is_fraud',
    'payment_type': 'transaction_type',
    'session_length_in_minutes': 'session_length_minutes',
}
BAF_AS_IS = [
    'email_is_free', 'employment_status', 'housing_status', 'phone_home_valid', 'phone_mobile_valid',
    'bank_months_count', 'has_other_cards', 'proposed_credit_limit', 'foreign_request', 'source',
    'device_os', 'keep_alive_session', 'device_distinct_emails_8w', 'device_fraud_count', 'zip_count_4w',
    'velocity_6h', 'velocity_24h', 'velocity_4w', 'bank_branch_count_8w', 'date_of_birth_distinct_emails_4w',
    'credit_risk_score', 'customer_age', 'days_since_request', 'intended_balcon_amount', 'current_address_months_count',
    'prev_address_months_count', 'name_email_similarity', 'income', 'month', 'x1', 'x2'
]

PAYSIM_RENAME = {
    'type': 'transaction_type',
    'amount': 'amount',
    'isFraud': 'is_fraud',
    'nameDest': 'counterparty_id',
    'oldbalanceOrg': 'orig_balance_before',
    'newbalanceOrig': 'orig_balance_after',
    'oldbalanceDest': 'dest_balance_before',
    'newbalanceDest': 'dest_balance_after',
    # Derived
    'step': 'event_time_ts',
}
PAYSIM_AS_IS = ['nameOrig', 'isFlaggedFraud', 'step']

SPARKOV_RENAME = {
    'trans_num': 'transaction_id',
    'unix_time': 'event_time_ts',
    'category': 'transaction_type',
    'amt': 'amount',
    'merchant': 'merchant_name',
    'merch_lat': 'merchant_lat',
    'merch_long': 'merchant_long',
    'is_fraud': 'is_fraud',
}
SPARKOV_AS_IS = ['cc_num', 'first', 'last', 'gender', 'street', 'city', 'state', 'zip', 'lat', 'long', 'city_pop', 'job', 'dob', 'trans_date_trans_time']

DGUARD_RENAME = {
    'uuid': 'transaction_id',
    'merchant_clean_name': 'merchant_name',
    'operation_type': 'transaction_type',
    'fraud_score': 'risk_score',
    'is_suspicious': 'risk_flag',
    # Derived passthrough of raw time string
    'operation_date': 'event_time_str',
}
DGUARD_AS_IS = ['amount', 'currency', 'description', 'categories', 'balance', 'user_id', 'account_id']


def _collect_mappings() -> Dict[str, List[Tuple[str, str]]]:
    """Return a dict: unified_col -> list of (source, original_name) pairs."""
    result: Dict[str, List[Tuple[str, str]]] = {}

    def add(source: str, ren: Dict[str, str], as_is: List[str]) -> None:
        for orig, uni in ren.items():
            result.setdefault(uni, []).append((source, orig))
        for col in as_is:
            result.setdefault(col, []).append((source, col))

    add('ieee', IEEE_RENAME, IEEE_AS_IS)
    add('ulb', ULB_RENAME, ULB_AS_IS)
    add('baf', BAF_RENAME, BAF_AS_IS)
    add('paysim', PAYSIM_RENAME, PAYSIM_AS_IS)
    add('sparkov_tx', SPARKOV_RENAME, SPARKOV_AS_IS)
    add('dguard_tx', DGUARD_RENAME, DGUARD_AS_IS)
    return result


def write_schema_csv(out_path: str | Path) -> Path:
    """Write a schema dictionary CSV with columns: column, pg_type, sources, origins.

    - sources: comma-separated source keys where the column appears
    - origins: semicolon-separated pairs like "source:original"
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    mappings = _collect_mappings()

    # Collect all columns either from declared schema or observed mappings
    all_columns = set(UNIFIED_SCHEMA_PG.keys()) | set(mappings.keys())

    with out.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['column', 'pg_type', 'sources', 'origins'])
        for col in sorted(all_columns):
            pg_type = UNIFIED_SCHEMA_PG.get(col, 'TEXT')
            origins = mappings.get(col, [])
            srcs = sorted({s for s, _ in origins})
            origins_str = '; '.join([f'{s}:{o}' for s, o in origins]) if origins else ''
            writer.writerow([col, pg_type, ','.join(srcs), origins_str])

    return out

