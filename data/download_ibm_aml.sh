#!/bin/bash
# Download the IBM Transactions for Anti-Money Laundering dataset (spec §6.1) via kagglehub.
#
# Public, synthetic, labelled, no PII. Raw data is gitignored and never committed (spec §6).
# This PUBLIC dataset downloads ANONYMOUSLY through kagglehub — no Kaggle token / kaggle.json needed.
# Kaggle dataset: ealtman2019/ibm-transactions-for-anti-money-laundering-aml
#
# We pull only the LI-Small files (transactions + patterns); the full dataset (all HI/LI variants)
# is multiple GB. The transactions CSV is ~620 MB. Files are cached under
# data/raw/kagglehub/... and symlinked to data/raw/<name> (the path ml/train.py expects).
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

# This cluster TLS-intercepts outbound HTTPS; point Python at the system CA bundle.
CA="${AEGIS_CA_BUNDLE:-/etc/ssl/certs/ca-certificates.crt}"
[ -f "$CA" ] && export REQUESTS_CA_BUNDLE="$CA" SSL_CERT_FILE="$CA"
export KAGGLEHUB_CACHE="${KAGGLEHUB_CACHE:-$PWD/data/raw/kagglehub}"

[ -d env ] && source env/bin/activate || true
python -c "import kagglehub" 2>/dev/null || pip install -q kagglehub

python - <<'PY'
import kagglehub, os, pathlib
ds = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
raw = pathlib.Path("data/raw"); raw.mkdir(parents=True, exist_ok=True)
# Add HI-/Medium variants here later if needed (spec §6.1: start LI-Small, scale to LI-Medium).
for fname in ["LI-Small_Trans.csv", "LI-Small_Patterns.txt"]:
    p = kagglehub.dataset_download(ds, path=fname)
    link = raw / fname
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(p)
    print(f"OK {fname}: {os.path.getsize(p)/1e6:.1f} MB  ->  data/raw/{fname}")
print("done — raw data stays under data/raw/ (gitignored), never committed.")
PY
