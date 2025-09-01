### Modeling Feature Candidates Requiring Retraining

Scope
- Track engineered features that are (a) available or easy to add in serving/ETL, (b) not part of the current trained model, and (c) would require retraining to take advantage of them for model lift.

Notes
- Adding these to responses or rules is safe without retraining, but including them as model inputs requires retraining (and re‑calibrating thresholds).
- Prefer time‑aware CV and change‑set ablations to quantify ROI before promoting any feature into production training.

1) is_weekend
- Definition: 1 if event day is Saturday/Sunday, else 0 (Mongo `$dayOfWeek`).
- Status: Implemented in service aggregation; not used by current model.
- Rationale: Weekend patterns differ for legitimate vs fraudulent activity.
- Retraining impact: Low complexity; include as a categorical/numeric feature; check for interaction with hour_of_day.

2) is_holiday
- Definition: 1 if `event_date` is present in an artifact list (`holiday_dates.json`), else 0.
- Status: Artifact support wired; feature flag computed post‑query when artifact present; not in model.
- Rationale: Holiday shopping/travel spikes; elevated fraud attempts.
- Retraining impact: Requires curated regional lists; evaluate per‑tenant/domain.

3) merchant_freq_global
- Definition: P(merchant) prior = global relative transaction frequency for normalized merchant name.
- Status: Generated and loaded (`merchant_freq_map.json`); attached to streamed features; not in model.
- Rationale: Rare merchants correlate with novelty risk; complements per‑user novelty.
- Retraining impact: Numeric input; consider log or rank scaling.

4) currency_normalized_amount
- Definition: Amount converted to a base currency (e.g., USD) using daily FX.
- Status: Not implemented; candidate as parallel field.
- Rationale: Cross‑currency comparability improves amount‑based signals.
- Retraining impact: Requires FX source and backfill; evaluate replacing or augmenting raw amount.

5) calendar seasonality (month/quarter sin/cos)
- Definition: Cyclic encodings for month of year, week of year.
- Status: Not implemented; easy to add similarly to hour_sin/cos.
- Rationale: Seasonality influences legitimate behavior and fraud attempts.
- Retraining impact: Low; include cautiously to avoid overfitting to fixed periods.

6) per‑merchant historical share (account‑level)
- Definition: Share of this merchant in last 30/90 days for the account.
- Status: Not implemented; requires additional windows per (partition_key, merchant).
- Rationale: Out‑of‑habit spend at a merchant is more anomalous.
- Retraining impact: Medium; adds stateful features; validate latency/compute budget.

7) balance depletion / velocity extensions
- Definition: Sum of debits / starting balance (24h), rolling std/mean of amounts.
- Status: Partially discussed; not fully implemented.
- Rationale: Cash‑out behavior and burstiness indicators.
- Retraining impact: Medium; ensure correctness of balance semantics in source.

Evaluation Plan
- Add candidate(s) to offline training pipeline; run time‑aware CV and P@k on target alert budgets.
- Check drift/availability across domains (ULB/IEEE vs DGuard) to prevent domain leakage.
- Update thresholds and SHAP global importance; document in `fraud_model_progress.md`.

Promotion Checklist
- [ ] Feature computed identically offline and online
- [ ] Latency budget not exceeded (aggregation windows)
- [ ] Time‑aware CV lift ≥ agreed threshold (e.g., +1–2% P@k at target k)
- [ ] Thresholds recalibrated; artifacts exported
- [ ] Service updated; deployment and smoke tests pass

