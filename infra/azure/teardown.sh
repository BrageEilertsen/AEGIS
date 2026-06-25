#!/usr/bin/env bash
# Remove all AEGIS Azure resources (stops all billing) by deleting the resource group.
set -euo pipefail
RG="${AEGIS_RG:-aegis-rg}"
echo "Deleting resource group '$RG' and everything in it…"
az group delete -n "$RG" --yes --no-wait
echo "Submitted. Resources are being removed in the background; billing stops as they delete."
