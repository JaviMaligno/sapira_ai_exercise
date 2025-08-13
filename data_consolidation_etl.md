## Unified Fraud Data Normalization and ETL Plan

This document describes how we will normalize and consolidate multiple heterogeneous sources into a single, wide, analysis-ready table for fraud detection experiments in this repository. We explicitly preserve provenance and allow nulls where fields don’t exist in all sources.

### In-Scope Sources

- Bank Account Fraud (BAF) Dataset Suite – NeurIPS 2022
  - Path: `/home/javier/repos/datasets/Bank Account Fraud (BAF) Dataset Suite - NeurIPS 2022/`
  - Files: `Base.csv`, `Variant I.csv`, `Variant II.csv`, `Variant III.csv`, `Variant IV.csv`, `Variant V.csv`
  - Label: `fraud_bool`
  - Nature: Application/behavior aggregates, device/channel, demographics

- IEEE-CIS Fraud Detection
  - Path: `/home/javier/repos/datasets/ieee-fraud-detection/`
  - Files: `train_transaction.csv` (+ `train_identity.csv`), `test_transaction.csv` (+ `test_identity.csv`)
  - Label: `isFraud` (train only)
  - Nature: Transaction + identity/device + many engineered features

- PaySim Synthetic Mobile Money
  - Path: `/home/javier/repos/datasets/PaySim Synthetic Mobile Money Dataset/PS_20174392719_1491204439457_log.csv`
  - Label: `isFraud` (and `isFlaggedFraud`)
  - Nature: Transfer type + pre/post balances for origin/destination

- ULB Credit Card Fraud (European Cardholders)
  - Path: `/home/javier/repos/datasets/ULB Credit Card Fraud Dataset (European Cardholders)/`
  - Files: `fraudTrain.csv`, `fraudTest.csv`
  - Label: `is_fraud`
  - Nature: Transaction + merchant + rich customer PII

- Sparkov Data Generation (Synthetic generator)
  - Path: `/home/javier/repos/Sparkov_Data_Generation/`
  - Customer schema: `data/customers.csv` (`|` delimited)
    - Headers: `ssn|cc_num|first|last|gender|street|city|state|zip|lat|long|city_pop|job|dob|acct_num|profile`
  - Transaction generator output schema (combined customer + transaction, `|` delimited)
    - Customer headers above plus transaction headers: `trans_num|trans_date|trans_time|unix_time|category|amt|is_fraud|merchant|merch_lat|merch_long`
  - Label: `is_fraud`

- DGuard MongoDB (Banking platform exploration)
  - Path: `/home/javier/repos/mongodb-exploration/`
  - Connection: see `README.md` for URI and auth
  - Database name: `dguard_transactions` (updated)
  - Key collection for modeling: `bank_transactions`
    - Fields include: `operation_date`, `amount`, `balance`, `currency`, `description`, `operation_type`, `fraud_score`, `is_suspicious`, `fraud_status`, `merchant_clean_name`, `categories`, `transfer_details{account_number, sender_receiver}`, `account_id`, `user_id`, `uuid`
  - Label: none; provides fraud scoring signals (`fraud_score`, `is_suspicious`)

### PII and Sensitive Data

- Contains direct PII: ULB, Sparkov (names, address, DOB, job, `cc_num`, `ssn`), DGuard (accounts, names within owners, KYC docs).
- Pseudonymous/derived: IEEE (card1–6, email domains, device), BAF (demographics/flags), PaySim (synthetic IDs only).
- Policy: hash or drop direct identifiers before model training; keep raw for lookup only in a separate, access-controlled store.

## Unified Schema (wide, nullable)

We create a denormalized, columnar table. Columns are grouped logically. Any field not present in a source remains null.

- Provenance and labels
  - `dataset` string (e.g., `baf_base`, `baf_var1`, `ieee_train`, `ieee_test`, `paysim`, `ulb_train`, `ulb_test`, `sparkov_tx`, `dguard_tx`)
  - `source_row_id` string (original PK or row index)
  - `transaction_id` string (IEEE `TransactionID`, ULB `trans_num`, DGuard `uuid`, PaySim synthetic id, else null)
  - `is_fraud` int8 nullable (BAF/IEEE/PaySim/ULB/Sparkov)
  - `risk_score` float32 nullable (DGuard `fraud_score`)
  - `risk_flag` int8 nullable (DGuard `is_suspicious`)
  - `risk_status` string nullable (DGuard `fraud_status`)

- Transaction core
  - `event_time_ts` int64 epoch seconds (IEEE `TransactionDT` mapped; ULB `unix_time`; PaySim `step*3600+epoch0`; DGuard `operation_date` parsed)
  - `event_time_str` string (ULB `trans_date_trans_time`, DGuard `operation_date` ISO, else null)
  - `amount` float32 (IEEE `TransactionAmt`, ULB `amt`, PaySim `amount`, DGuard `amount`)
  - `currency` string (DGuard `currency`)
  - `transaction_type` string (ULB `category`, PaySim `type`, BAF `payment_type`, DGuard `operation_type`)
  - `product_code` string (IEEE `ProductCD`)
  - `channel_source` string (BAF `source`)

- Merchant / counterparty
  - `merchant_name` string (ULB `merchant`, Sparkov `merchant`, DGuard `merchant_clean_name`)
  - `merchant_lat` float32, `merchant_long` float32 (ULB/Sparkov)
  - `counterparty_id` string (PaySim `nameDest`, DGuard `transfer_details.sender_receiver`)

- Payment instrument
  - `card_brand` string (IEEE `card4`)
  - `card_type` string (IEEE `card6`)
  - `card_token_hash` string (hash of IEEE `card1` or ULB/Sparkov `cc_num`)

- Location and distance
  - `addr1` int16, `addr2` int16, `dist1` float32, `dist2` float32 (IEEE)
  - `customer_city`, `customer_state`, `customer_zip`, `customer_lat`, `customer_long`, `customer_city_pop` (ULB/Sparkov)

- Device / identity / email
  - `device_type`, `device_info` (IEEE identity)
  - `device_os` (BAF)
  - `email_domain_payer`, `email_domain_receiver` (IEEE `P_emaildomain`, `R_emaildomain`)

- Customer/account metadata
  - `customer_id_hash` string (hash of ULB/Sparkov PII, DGuard `user_id` hashed)
  - `account_id_hash` string (DGuard `account_id` hashed)
  - `customer_age` int16 (BAF), `dob` date (ULB/Sparkov), `gender` string (ULB/Sparkov)
  - `employment_status`, `housing_status`, `income` (BAF)
  - `bank_months_count` int16, `has_other_cards` int8 (BAF)

- Balances (PaySim)
  - `orig_balance_before`, `orig_balance_after`, `dest_balance_before`, `dest_balance_after`

- Behavioral / aggregates (BAF)
  - `velocity_6h`, `velocity_24h`, `velocity_4w`, `zip_count_4w`, `bank_branch_count_8w`, `device_distinct_emails_8w`, `device_fraud_count`,
    `credit_risk_score`, `proposed_credit_limit`, `keep_alive_session`, `session_length_minutes`, `date_of_birth_distinct_emails_4w`,
    `days_since_request`, `month`, `baf_x1`, `baf_x2`

- IEEE engineered and identity blocks (namespaced)
  - `ieee_C1..C14`, `ieee_D1..D15`, `ieee_M1..M9`, `ieee_V1..V339`, `id_01..id_38`

Notes
- Keep `dataset` to preserve domain differences. Avoid naïvely mixing domains for a single model without domain features or adaptation.
- Many columns will be sparsely populated; this is expected.

## Dataset-Specific Mappings

Below, “→” indicates target column.

### BAF (Base, Variant I–V)

- Label: `fraud_bool` → `is_fraud`
- Time: `days_since_request` → keep; `month` → keep; `event_time_ts` null
- Amount: `intended_balcon_amount`, `proposed_credit_limit` → keep in aggregates; `amount` null
- Type/Channel: `payment_type` → `transaction_type`, `source` → `channel_source`
- Device/Email: `device_os` → `device_os`, `device_distinct_emails_8w`, `device_fraud_count` → aggregates
- Demographics: `customer_age`, `income`, `employment_status`, `housing_status`, `email_is_free` (keep)
- Behavior: `zip_count_4w`, `velocity_*`, `bank_branch_count_8w`, `date_of_birth_distinct_emails_4w`, `keep_alive_session`, `session_length_in_minutes`
- Variant-only: `x1`, `x2` → `baf_x1`, `baf_x2`

### IEEE-CIS (train/test + identity)

- Join: `*_transaction.csv` LEFT JOIN `*_identity.csv` on `TransactionID`
- Label: `isFraud` → `is_fraud` (test null)
- IDs: `TransactionID` → `transaction_id`
- Time: `TransactionDT` → `event_time_ts_raw` (store raw seconds), optionally map to epoch (commonly 2017-12-01). Keep both.
- Amount: `TransactionAmt` → `amount`
- Product/instrument: `ProductCD` → `product_code`, `card4` → `card_brand`, `card6` → `card_type`, `card1` (hash) → `card_token_hash`
- Email: `P_emaildomain` → `email_domain_payer`, `R_emaildomain` → `email_domain_receiver`
- Location/dist: `addr1`, `addr2`, `dist1`, `dist2`
- Device: `DeviceType` → `device_type`, `DeviceInfo` → `device_info`
- Engineered blocks: prefix to `ieee_*` and keep as-is (`C*`, `D*`, `M*`, `V*`), identity `id_01..id_38`

### PaySim

- Label: `isFraud` → `is_fraud`; also retain `isFlaggedFraud`
- Time: `step` (hours since start) → `event_time_ts = epoch0 + step*3600`
- Amount: `amount` → `amount`
- Type: `type` → `transaction_type`
- Counterparties: `nameOrig` (hash) → `customer_id_hash`, `nameDest` → `counterparty_id`
- Balances: map `oldbalanceOrg` → `orig_balance_before`, `newbalanceOrg` → `orig_balance_after`, `oldbalanceDest` → `dest_balance_before`, `newbalanceDest` → `dest_balance_after`

### ULB (fraudTrain/fraudTest)

- Label: `is_fraud` → `is_fraud`
- IDs: `trans_num` → `transaction_id`
- Time: `unix_time` → `event_time_ts`, `trans_date_trans_time` → `event_time_str`
- Amount: `amt` → `amount`
- Type/Merchant: `category` → `transaction_type`, `merchant` → `merchant_name`, `merch_lat/merch_long` → merchant coords
- Customer PII: `cc_num` (hash) → `card_token_hash`, `first/last/street/city/state/zip/lat/long/city_pop/job/dob` → mapped customer fields; also compute `customer_id_hash` from stable PII subset

### Sparkov synthetic (customers + generated transactions)

- Transactions contain both customer and transaction attributes (pipe-delimited)
- Label: `is_fraud` → `is_fraud`
- IDs: `trans_num` → `transaction_id`
- Time: `unix_time` → `event_time_ts`; also `trans_date`/`trans_time` → `event_time_str`
- Amount/Type/Merchant: `amt`, `category` → `transaction_type`, `merchant` → `merchant_name`, `merch_lat/merch_long`
- Customer PII: `cc_num` (hash) → `card_token_hash`; compute `customer_id_hash`; map `city/state/zip/lat/long/city_pop/job/dob/gender`

### DGuard MongoDB (`bank_transactions` in `dguard_transactions`)

- Label: none; keep risk signals
- IDs: `uuid` → `transaction_id`, `user_id` (hash) → `customer_id_hash`, `account_id` (hash) → `account_id_hash`
- Time: `operation_date` (ISO) → parse to `event_time_ts` and keep as `event_time_str`
- Amount/Currency: `amount` → `amount` (sign indicates debit/credit), `currency`
- Type/Merchant: `operation_type` → `transaction_type`, `merchant_clean_name` → `merchant_name`
- Fraud signals: `fraud_score` → `risk_score`, `is_suspicious` → `risk_flag`, `fraud_status` → `risk_status`
- Other: `balance` → keep as account-level balance at time of transaction; `categories` (JSON) → optional normalized tag list

## ETL Procedure

### 1) Environment

- Use Python 3.10+
- Create venv in `sapira_ai_exercise/.venv` and install:
  - `pandas`, `pyarrow`, `numpy`
  - `polars` (optional, speed), `dask[dataframe]` (optional, scale)
  - `python-dateutil`, `tqdm`
  - For Mongo: `pymongo`

### 2) Reading files and joins

- BAF: read CSVs with default comma delimiter; unify shared columns; add `dataset` per file
- IEEE: read `train_transaction.csv`; left-join `train_identity.csv` by `TransactionID`; same for test; add `dataset`
- PaySim: read CSV; compute `event_time_ts` by `epoch0 + step*3600` (choose `epoch0 = 2019-01-01T00:00:00Z` unless otherwise specified)
- ULB: read train/test; add `dataset` and map columns
- Sparkov: read generated transactions (pipe `|` delimiter). If only `customers.csv` exists, run generator to produce transactions before ETL
- Mongo (DGuard): connect and read `bank_transactions` projection of required fields; page via cursor; convert to DataFrame

### 3) Transformations

- Standardize dtypes: numeric to float32/int32 where sensible; booleans to int8 {0,1}
- Time: compute `event_time_ts` (UTC, seconds); keep a human-readable `event_time_str` when present; for IEEE keep `TransactionDT` raw as `event_time_ts_raw`
- Hashing: compute stable SHA-256 for PII (`card_token_hash`, `customer_id_hash`, `account_id_hash`) with a project salt
- Normalization: trim strings, lowercase domains/types, standardize categories
- Namespacing: prefix IEEE engineered fields to avoid clashes
- Provenance: set `dataset`, `source_row_id`

### 4) Output

- Write unified table as Parquet partitioned by `dataset` to `sapira_ai_exercise/data/unified/`
- Also emit a CSV schema dictionary describing each column and its origins: `sapira_ai_exercise/data/unified/schema.csv`

## Minimal Reference Code Snippets

```python
# venv activation and requirements omitted here for brevity
import pandas as pd
import numpy as np
from hashlib import sha256

def h(x, salt):
    if pd.isna(x):
        return np.nan
    return sha256((salt + str(x)).encode()).hexdigest()

# Example: IEEE train load + identity join
tx = pd.read_csv('/home/javier/repos/datasets/ieee-fraud-detection/train_transaction.csv')
idn = pd.read_csv('/home/javier/repos/datasets/ieee-fraud-detection/train_identity.csv')
df = tx.merge(idn, on='TransactionID', how='left')
df.rename(columns={'TransactionID':'transaction_id', 'isFraud':'is_fraud', 'TransactionAmt':'amount',
                   'ProductCD':'product_code', 'P_emaildomain':'email_domain_payer',
                   'R_emaildomain':'email_domain_receiver', 'DeviceType':'device_type',
                   'DeviceInfo':'device_info'}, inplace=True)
df['dataset'] = 'ieee_train'
df['card_token_hash'] = df['card1'].apply(lambda v: h(v, 'SALT'))
```

```python
# Example: Sparkov transactions (pipe-delimited)
sparkov = pd.read_csv('/path/to/generated_transactions.psv', sep='|')
sparkov.rename(columns={'trans_num':'transaction_id','unix_time':'event_time_ts',
                        'category':'transaction_type','amt':'amount','merchant':'merchant_name',
                        'merch_lat':'merchant_lat','merch_long':'merchant_long',
                        'is_fraud':'is_fraud'}, inplace=True)
sparkov['card_token_hash'] = sparkov['cc_num'].apply(lambda v: h(v, 'SALT'))
sparkov['dataset'] = 'sparkov_tx'
```

```python
# Example: Mongo (DGuard) bank_transactions
from pymongo import MongoClient
client = MongoClient('mongodb://...')  # see mongodb-exploration/README.md
coll = client['dguard_transactions']['bank_transactions']
fields = {
  'uuid':1,'user_id':1,'account_id':1,'operation_date':1,'amount':1,'currency':1,
  'description':1,'operation_type':1,'fraud_score':1,'is_suspicious':1,'fraud_status':1,
  'merchant_clean_name':1,'categories':1,'balance':1
}
docs = list(coll.find({}, fields))
dg = pd.DataFrame(docs)
dg.rename(columns={'uuid':'transaction_id','merchant_clean_name':'merchant_name',
                   'operation_type':'transaction_type','fraud_score':'risk_score',
                   'is_suspicious':'risk_flag'}, inplace=True)
dg['dataset'] = 'dguard_tx'
```

## Overlaps and Gaps (high level)

- Common across most: `is_fraud` (except DGuard test-time signals), `amount`, some notion of time
- Available in subsets:
  - Card/instrument: IEEE, ULB, Sparkov
  - Merchant info: ULB, Sparkov, DGuard; not in BAF/IEEE
  - Device/email: IEEE, BAF
  - Demographics: BAF, ULB, Sparkov
  - Balances: PaySim only

## Quality and Caveats

- Domain shift: BAF (application risk) vs IEEE/ULB/Sparkov/PaySim (transaction logs) vs DGuard (bank ops + risk). Use `dataset` as a feature and/or train per-domain models.
- Label availability: IEEE test has no labels; DGuard has risk signals, not labels. Keep them but don’t train supervised on them.
- PII governance: hash or drop direct identifiers before modeling. Store salt securely.

## Deliverables

- Unified Parquet at `sapira_ai_exercise/data/unified/` partitioned by `dataset`
- Schema dictionary CSV with source mappings
- Optional: dataset-specific derived feature sets saved per domain



