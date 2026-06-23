#!/bin/bash
# Download the IBM Transactions for Anti-Money Laundering dataset (spec §6.1).
#
# Public, synthetic, labelled, no PII. Raw data is gitignored and never committed (spec §6).
# Start with the LI-Small variant (laundering ratio ~0.05-0.1%, realistic base rate); scale to
# LI-Medium once the pipeline is stable.
#
# Kaggle dataset: ealtman2019/ibm-transactions-for-anti-money-laundering-aml
# Files per variant: <variant>_Trans.csv (transactions), accounts CSV, and a patterns file
# enumerating the injected laundering typologies.
#
# Prerequisites:
#   - Kaggle CLI:  pip install kaggle
#   - API token:   ~/.kaggle/kaggle.json  (chmod 600), from your Kaggle account settings.
#
# Phase 0: placeholder. Flesh out + verify in Phase 1.
set -euo pipefail

# This cluster TLS-intercepts outbound HTTPS; point requests-based tools (kaggle) at the
# system CA bundle so certificate verification succeeds.
CA="${AEGIS_CA_BUNDLE:-/etc/ssl/certs/ca-certificates.crt}"
[ -f "$CA" ] && export REQUESTS_CA_BUNDLE="$CA" SSL_CERT_FILE="$CA"

RAW_DIR="$(cd "$(dirname "$0")" && pwd)/raw"
mkdir -p "$RAW_DIR"

echo "TODO (Phase 1): download IBM-AML LI-Small into $RAW_DIR, e.g.:"
echo "  kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml -p \"$RAW_DIR\""
echo "  unzip -n \"$RAW_DIR\"/*.zip -d \"$RAW_DIR\""
echo
echo "Then confirm LI-Small_Trans.csv (+ accounts/patterns files) are present."
echo "Raw data stays under data/raw/ and is gitignored — never commit it."
