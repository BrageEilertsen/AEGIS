#!/usr/bin/env bash
# One-command AEGIS deploy to Azure — Container Apps + PostgreSQL Flexible Server.
# Images are built server-side in your registry (az acr build), so NO local Docker is needed.
#
# Prereqs: Azure subscription + `az login` (see infra/azure/README.md). Override any of the
# AEGIS_* env vars below to customise.
set -euo pipefail

RG="${AEGIS_RG:-aegis-rg}"
LOCATION="${AEGIS_LOCATION:-norwayeast}"
PREFIX="${AEGIS_PREFIX:-aegis}"
ACR="${AEGIS_ACR:-aegisacr$(date +%s | tail -c 6)}"   # globally-unique, lowercase alphanumeric
TAG="${AEGIS_TAG:-latest}"
PG_PASSWORD="${AEGIS_PG_PASSWORD:-Aegis$(openssl rand -hex 12)9X}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "==> Ensuring resource providers are registered"
for ns in Microsoft.App Microsoft.ContainerRegistry Microsoft.DBforPostgreSQL Microsoft.OperationalInsights; do
  az provider register --namespace "$ns" -o none || true
done

echo "==> Resource group '$RG' in '$LOCATION'"
az group create -n "$RG" -l "$LOCATION" -o none

echo "==> Container registry '$ACR' (Basic)"
az acr create -n "$ACR" -g "$RG" --sku Basic --admin-enabled true -o none

echo "==> Building the three images in the registry (server-side; a few minutes)"
az acr build -r "$ACR" -t "aegis-inference:$TAG" --file inference/Dockerfile .
az acr build -r "$ACR" -t "aegis-api:$TAG"       --file Dockerfile api
az acr build -r "$ACR" -t "aegis-frontend:$TAG"  --file Dockerfile frontend

echo "==> Deploying infrastructure + apps (Bicep)"
az deployment group create -g "$RG" -f infra/azure/main.bicep \
  -p acrName="$ACR" imageTag="$TAG" pgAdminPassword="$PG_PASSWORD" -o none

FRONTEND=$(az deployment group show -g "$RG" -n main --query properties.outputs.frontendUrl.value -o tsv)
API=$(az deployment group show -g "$RG" -n main --query properties.outputs.apiUrl.value -o tsv)

cat <<EOF

===================================================================
 AEGIS deployed.
   Frontend (open this):  $FRONTEND
   API:                   $API
   Postgres admin pwd:    $PG_PASSWORD   (save it)
 First load wakes the apps + warms the model — give it a minute.
 Tear everything down with: infra/azure/teardown.sh
===================================================================
EOF
