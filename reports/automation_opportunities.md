### Automation Opportunities (Artifacts and Jobs)

1) Operation type synonyms management
- Source: `reports/phase2/ulb_gbdt/operation_type_synonyms.json`
- Opportunity: nightly/job-based sync to allow domain experts to edit a central file (e.g., S3) and auto-reload.
- Implementation:
  - Store authoritative JSON in S3; add a scheduled fetch at service startup and a lightweight `/v1/reload-artifacts` admin endpoint.
  - Optional validation job to catch duplicates and conflicting synonyms.

2) Merchant frequency map refresh
- Source: Mongo `bank_transactions`
- Opportunity: schedule a daily/weekly compute for `merchant_freq_map.json` with rolling 90-day window.
- Implementation:
  - Scripted aggregation to compute counts and relative frequency; write to artifacts dir/S3.
  - Post-write checksum/version file; service hot-reload endpoint or rolling restart to apply.

3) Category (operation type) robust stats
- Artifact(s): medians/IQR per category (used for `amount_z_iqr_cat`), p99.5 caps for winsorization.
- Opportunity: nightly recompute into a `fraud_params` collection and/or JSON artifacts.
- Implementation:
  - Mongo aggregation over trailing window per `operation_type_canonical` to compute median, IQR, p99.5.
  - Export to `fraud_params` and JSON. Add versioning and checksum.

4) Thresholds recalibration
- Artifacts: per-category thresholds JSON (e.g., 0.5%, 1%).
- Opportunity: periodic recalibration using shadow/label feedback and alert budget targets.
- Implementation:
  - Batch job computes precision@k and updates thresholds; write new JSON with version tag.
  - Approval gate before swapping live thresholds; audit log of changes.

5) Drift monitoring snapshots
- Artifacts: weekly quantiles/missingness reports for key features.
- Opportunity: scheduled export with alerts when deviations exceed tolerances.
- Implementation:
  - Compute and store quantiles; compare vs baseline; push alerts (Slack/email) when drift detected.

6) Artifact loader hot-reload
- Opportunity: avoid service restarts to pick up artifact changes.
- Implementation:
  - Add `/v1/reload-artifacts` admin endpoint guarded by role/secret; reload on demand.
  - Optional file watcher in container with debounce.

7) CI integration
- Opportunity: validate artifact schema and presence during PR (e.g., synonyms and thresholds).
- Implementation:
  - Add CI step to lint JSON artifacts and ensure required keys exist; block merge if invalid.

8) FX rates refresh
- Source: external provider (e.g., ECB-daily, exchangerate.host, OpenExchangeRates) → `fx_rates.json` in artifacts.
- Opportunity: daily fetch of base→USD (or chosen base) rates. Avoid manual edits.
- Implementation:
  - Scheduled job fetches latest rates, writes `{ "USD": 1.0, "EUR": 1.08, ... }` with version stamp.
  - Validate presence of required currencies observed in data; alert if missing.
  - Hot-reload endpoint or rolling restart to apply.

Notes
- Keep all automated jobs idempotent and versioned; retain last N versions for rollback.
- Prefer rolling windows (e.g., trailing 90 days) for stability and recency.

