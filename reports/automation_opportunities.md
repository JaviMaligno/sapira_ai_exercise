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

9) MLflow experiment tracking and model registry
- Source: training scripts in `src/fraud_mvp/*.py` and notebooks
- Opportunity: replace manual S3 uploads with tracked, versioned models and artifacts.
- Implementation:
  - Stand up MLflow server (S3/MinIO for artifact store, SQLite/Postgres for backend DB).
  - Instrument training code to `mlflow.start_run()`; log params/metrics; log artifacts (pipelines, thresholds, rule params, explainability).
  - Register models in MLflow Registry with stages (None → Staging → Production); enforce approvals via code owners.

10) Automated export from MLflow Registry to serving artifacts
- Source: MLflow model version events (promoted to Production)
- Opportunity: immutable, versioned artifact bundles consumable by the service.
- Implementation:
  - Export `pipeline.pkl`, `isotonic.pkl`, `if_pipe.pkl`, `*per_category_thresholds_*.json`, `rule_params.json`, `operation_type_synonyms.json`, `merchant_freq_map.json`, optional `fx_rates.json`.
  - Write a `manifest.json` capturing: `model_name`, `model_version`, `mlflow_run_id`, `git_sha`, `trained_at`, `data_snapshot_id`, `metrics`.
  - Layout: `s3://<bucket>/fraud/artifacts/ulb_gbdt/{version}/...` and a pointer `s3://.../ulb_gbdt/current.json` → {version}.
  - Scoring service reads `/artifacts/<version>` or `/artifacts/current` symlink; uses `/v1/reload-artifacts` to hot-swap.

11) CI/CD integration (GitHub Actions)
- Opportunity: consistent train→evaluate→register→promote→export→deploy pipeline with quality gates.
- Implementation:
  - On PR: ruff lint, unit/integration tests, artifact schema lint, small smoke training run with metrics, block on failures.
  - On main: full training; compare against champion (AUPRC, precision@k, latency budget); if pass, auto-register to Staging.
  - Staging validation job: shadow scoring on held-out/stream sample; if pass, promote to Production; trigger export and service reload.
  - Embed version tags (semver + git SHA) into artifact path and Docker image labels.

12) Artifact schema validation (expand of item 7)
- Opportunity: prevent runtime issues from malformed JSONs or missing keys.
- Implementation:
  - Define JSONSchema for `operation_type_synonyms.json`, thresholds files, `fraud_params` export, `fx_rates.json` (base_currency, last_updated, exchange_rates map), `manifest.json`.
  - Add a `poetry` script to validate; run in CI and pre-deploy.

13) Data quality and snapshot versioning
- Opportunity: reproducible training and safer retraining.
- Implementation:
  - Great Expectations/Pandera checks for ETL outputs (row counts, missingness, value ranges, allowed categories).
  - Persist a `data_snapshot_id` (e.g., date range + content hash) and log it to MLflow runs and `manifest.json`.
  - Block training/promotion on failed checks.

14) ETL orchestration and scheduled artifacts
- Source: `sapira_etl` and Mongo aggregations
- Opportunity: automate regular recomputes and keep derived artifacts fresh.
- Implementation:
  - Orchestrate nightly/weekly jobs (cron/Airflow): merchant frequency (item 2), category robust stats (item 3), thresholds (item 4), FX (item 8).
  - Enforce idempotency and windowed aggregation; emit `version.json` per artifact with checksum.

15) Drift monitoring and retraining triggers
- Opportunity: respond to concept drift without manual intervention.
- Implementation:
  - Compute PSI/KL for key features and score rank stability vs. training baseline; alert on breach.
  - If sustained drift N days and alert budget pressure detected, trigger retraining pipeline; gate by offline metrics before promotion.

16) Shadow/canary deployment automation
- Opportunity: safer rollouts for new models.
- Implementation:
  - Feature flag or traffic-split config to route 1–5% to the new artifact version; compare online metrics.
  - Auto-rollback on thresholds (FP rate +10%, latency +20ms, alert rate surge); promote on stable performance.

17) Model provenance and `/v1/metadata`
- Opportunity: transparent lineage in serving.
- Implementation:
  - Serve `model_version`, `mlflow_run_id`, `git_sha`, `artifact_checksums`, `fx_last_updated` via `/v1/metadata`.
  - Log correlation id, artifact version, and thresholds version with each request for auditability.

18) Standardized artifact layout and atomic swap
- Opportunity: predictable ops and zero-downtime updates.
- Implementation:
  - Directory strategy: `/artifacts/ulb_gbdt/{YYYYMMDD-HHMMSS}-{semver}-{sha}/...` + `/artifacts/ulb_gbdt/current` symlink.
  - Deploy step updates pointer atomically; call `/v1/reload-artifacts`.

19) File-watcher hot-reload (optional)
- Opportunity: eliminate even the minimal admin step when artifacts update.
- Implementation:
  - Lightweight inotify-based watcher in container; debounce; reload when `manifest.json` checksum changes.

20) Metrics and alerting automation
- Opportunity: actionable observability without manual checks.
- Implementation:
  - Prometheus scraping for Mongo agg latency, scoring latency, requests, alert rate, per-category thresholds hit.
  - SLO alerts (e.g., P95 scoring latency, daily alert volume bounds); dashboard templates.

21) Security and governance for artifact operations
- Opportunity: protect reloads and promotions.
- Implementation:
  - Guard `/v1/reload-artifacts` by admin scope; sign and verify `manifest.json`; audit log promotions with actor and reason.

22) Automated model card and documentation
- Opportunity: consistent documentation per release.
- Implementation:
  - Generate a model card Markdown from MLflow params/metrics and artifacts; include top features/SHAP plots; store under `reports/phase2/...` and link from `PROGRESS.md`.

23) Lightweight feature registry
- Opportunity: reduce train-serve skew and drift of feature definitions.
- Implementation:
  - YAML/JSON feature specs (name, owner, logic, sources, null policy) in repo; versioned in git; validated pre-train and pre-deploy.

24) End-to-end release workflow
- Opportunity: single action to go from code to live model.
- Implementation:
  - Workflow steps: train → evaluate → register → promote → export → update pointer → reload → smoke `/v1/health` and `/v1/score` → update `PROGRESS.md` and `reports/fraud_model_progress.md` with run ids and metrics.

25) Artifact pinning per environment
- Opportunity: deterministic staging vs production.
- Implementation:
  - Use env var `ARTIFACTS_DIR=/artifacts/ulb_gbdt/{version}`; staging points to candidate; prod to `current`.

