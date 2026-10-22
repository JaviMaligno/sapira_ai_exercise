# Runtime security follow-up, 22 October 2026

This patch updates compatible dependencies and the Python/CA/pip bootstrap.
It does not complete remediation of the MLflow 2 dependency tree or the
independent MLflow 2.18.0 service installations in Compose and Terraform.
Those startup installs do not consume `poetry.lock`. Their existing service
version, basic-auth flags, registry and artifact/storage paths are preserved.
Do not interpret the project lock audit as a service image audit.

## Constraints requiring a separate migration

The resolved MLflow 2.22.5 metadata requires `pyarrow>=4,<20`; patched
PyArrow for GHSA-rgxp-2hwp-jwgg starts at 23.0.1. MLflow 2 also retains
security advisories requiring a supported 3.x release. Merely changing the
client's dependency to 3.x would alter the model logging contract.

The existing `mlflow[extras]` selects AzureML. Latest AzureML 1.61.0.post4
requires `pyopenssl>=25.3.0,<=26.0.0`, `paramiko>=2.0.8,<4`, `pathspec<1`
and `packaging>=20,<26`. PyOpenSSL 26.0.0 requires
`cryptography>=46,<47`; fixes in later cryptography branches therefore need
a compatible parent change. Paramiko GHSA-r374-rxx8-8654 has no fixed event
in the advisory snapshot and includes the permitted 3.x branch.

virtualenv 21.14.6 through 21.14.12 requires packaging>=26.3, incompatible with AzureML's
packaging<26 and MLflow-skinny 2's packaging<25. The lock uses the latest
resolvable version, 21.14.5, with residual advisories
reported separately. Environment creation for this patch uses corrected
uv 0.13.5, not that application virtualenv package.

Black 26.10.1 requires pathspec>=1, which cannot share an environment with
these Azure extras. `tooling/formatting/uv.lock` isolates the fixed formatter;
the application no longer declares or locks the old Black. Formatter updates
and its frozen check must use the nested project, not `poetry run black`.

## Dedicated MLflow migration plan

1. Choose a currently supported MLflow 3 release after a fresh public-package
   audit, not just the first version fixing one advisory. Resolve a client
   candidate separately. Inventory actual Azure tracking/artifact usage and
   obtain evidence before removing `extras`; retain all storage support until
   that decision is justified. Re-evaluate PyArrow, cryptography, Paramiko and
   virtualenv against the resulting parents. Do not override incompatible
   metadata.
2. Adapt `src/sapira_etl/mlflow_client.py` explicitly. `log_training_run`
   currently returns a run ID, discards the `ModelInfo` returned by
   `mlflow.sklearn.log_model(model, "model", serialization_format="pickle")`,
   and `register_model` builds `runs:/{run_id}/model`. MLflow 3 logged models
   may use a model ID URI. Capture the returned `ModelInfo.model_uri` and
   persist it as run metadata while keeping the existing public run-ID return
   value. Registration must prefer that captured URI and retain the legacy
   `runs:/` fallback for existing runs. Review stage transitions explicitly;
   aliases must not silently replace the current stage argument.
3. Prove that adaptation with mocked client contract tests, then a real
   isolated Linux roundtrip. Cover new and legacy runs, params/metrics/tags,
   metadata artifacts, logged-model URI selection, registration/version return,
   default stage and an explicit stage, missing model artifacts, and readback
   of a synthetic trusted sklearn model with predictions matching before/after.
   Keep model libraries and existing model blobs unchanged. Test client 2 /
   server 2 as baseline, candidate client 3 / server 2.18 for supported flows,
   and candidate client 3 / server 3. Explicitly reject unsupported matrix cells
   rather than claiming compatibility from import success.
4. Prepare disposable database and object-storage copies. Back up DB schema,
   tracking/auth data and artifact paths; record restore commands and old image
   digest. Run the documented schema upgrade only against the disposable copy.
   Verify anonymous denial, reader/writer/admin permissions, per-experiment
   and registered-model access, basic-auth behavior, uploads/downloads and
   existing artifact URIs. Confirm rollback by restoring the copy and starting
   the original service. No production DB, credential or auth migration belongs
   in this patch.
5. Replace both independent service installs with a reviewed reproducible
   service lock/image and the tested MLflow version together. Preserve the
   public ECR registry prefix in Terraform, S3 paths and service flags. CI must
   build/scan the Linux image, render a representative synthetic WeasyPrint
   PDF with Pango/fonts and compare layout, and run the focused client matrix
   before the normal Ready suite. Deploy only after review/CI and environment
   authorization; no Terraform apply or server startup was performed locally.

## Evidence limits

The personal repository's Dependabot endpoint returned HTTP 403. Public OSV
package/version queries are independent evidence, not proof that inaccessible
alerts are clear. The delivery receipt records exact remaining IDs and lock
versions, plus checks and checks deferred to Linux CI. Full MLflow imports,
HTTP/Docker/DB integration tests and tests reading real fraud datasets are
not substitute local smoke tests for this dependency patch.
