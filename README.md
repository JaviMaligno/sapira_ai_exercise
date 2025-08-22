# Sapira AI Exercise - Fraud Data ETL

Environment managed with Poetry. Use `docker-compose.yml` to run Postgres.

Commands:

```bash
poetry install --no-interaction
poetry run etl-run --parquet-out sapira_ai_exercise/data/unified
# with Postgres
poetry run etl-run --write-postgres --pg-table unified_transactions
```


Chunked/streaming ETL (default 100k rows per chunk):

```bash
# run Postgres locally
docker compose up -d postgres

# write to parquet + Postgres in chunks
poetry run etl-run --write-postgres --pg-table unified_transactions --chunk-rows 100000

# limit rows per source (testing) and control chunk size
poetry run etl-run --limit-rows 1000 --chunk-rows 500 --write-postgres --pg-table unified_transactions_test
```

Schema dictionary CSV:

```bash
# generate schema dictionary at sapira_ai_exercise/data/unified/schema.csv
poetry run python -c "from sapira_etl.schema import write_schema_csv; write_schema_csv('sapira_ai_exercise/data/unified/schema.csv')"
```

Notes:
- ETL supports parquet-only or Postgres writes. When writing to Postgres, ingestion uses COPY for large/wide chunks and handles schema evolution by adding missing TEXT columns as needed.
- Parquet output is partitioned by `dataset` with multiple `part-000N.parquet` files when chunking.

## Reports

- Model status and current architecture/training/metrics: `reports/model_status.md`


