#!/usr/bin/env bash
# Deploy AEGIS to Azure — Container Apps + PostgreSQL Flexible Server.
# Images are built by GitHub Actions and pulled from public GHCR, so there's NO local build and
# NO ACR (ACR Tasks are blocked on free subscriptions). See infra/azure/README.md.
#
# Before running: (1) the build-images GitHub Action has finished, and (2) the three GHCR packages
# (aegis-inference / aegis-api / aegis-frontend) are set to PUBLIC. Then: az login, then this.
set -euo pipefail

RG="${AEGIS_RG:-aegis-rg}"
LOCATION="${AEGIS_LOCATION:-norwayeast}"
OWNER="${AEGIS_GHCR_OWNER:-brageeilertsen}"
TAG="${AEGIS_TAG:-latest}"
PG_PASSWORD="${AEGIS_PG_PASSWORD:-Aegis$(openssl rand -hex 12)9X}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "==> Registering resource providers"
for ns in Microsoft.App Microsoft.DBforPostgreSQL Microsoft.OperationalInsights; do
  az provider register --namespace "$ns" -o none || true
done

echo "==> Resource group '$RG' in '$LOCATION'"
az group create -n "$RG" -l "$LOCATION" -o none

echo "==> Deploying (images: ghcr.io/$OWNER/aegis-*:$TAG)"
az deployment group create -g "$RG" -f infra/azure/main.bicep \
  -p pgAdminPassword="$PG_PASSWORD" \
     inferenceImage="ghcr.io/$OWNER/aegis-inference:$TAG" \
     apiImage="ghcr.io/$OWNER/aegis-api:$TAG" \
     frontendImage="ghcr.io/$OWNER/aegis-frontend:$TAG" \
  -o none

FRONTEND=$(az deployment group show -g "$RG" -n main --query properties.outputs.frontendUrl.value -o tsv)
API=$(az deployment group show -g "$RG" -n main --query properties.outputs.apiUrl.value -o tsv)
cat <<EOF

===================================================================
 AEGIS deployed.
   Frontend (open this):  $FRONTEND
   API:                   $API
   Postgres admin pwd:    $PG_PASSWORD   (save it)
 First load wakes the apps + warms the model — give it a minute.
 Tear down with: infra/azure/teardown.sh
===================================================================
EOF
