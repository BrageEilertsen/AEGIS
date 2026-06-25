# Deploying AEGIS to Azure

A one-command deploy of the full stack to **Azure Container Apps** + **Azure Database for
PostgreSQL Flexible Server**, sized for the **Azure free account** ($200 credit / free tiers,
spending protection). Images build **server-side** in your registry — no local Docker needed.

## Prerequisites

1. An **Azure subscription** (the free account works: https://azure.microsoft.com/free).
2. The **Azure CLI**: `brew install azure-cli` (macOS) or https://aka.ms/InstallAzureCLI.
3. Log in and select your subscription:
   ```bash
   az login
   az account set --subscription "<your-subscription-id>"
   ```
   (The deploy script registers the required resource providers for you.)

## Deploy

```bash
./infra/azure/deploy.sh
```

That will: create a resource group + container registry, build the three images in the registry,
provision Postgres + the Container Apps environment, deploy the three apps, and print the **public
frontend URL**. First load wakes the apps and warms the model — give it ~a minute.

Customise via env vars, e.g.:
```bash
AEGIS_LOCATION=westeurope AEGIS_RG=aegis-demo ./infra/azure/deploy.sh
```

| Variable | Default | Notes |
|---|---|---|
| `AEGIS_LOCATION` | `norwayeast` | Any region with Container Apps + PG Flexible Server. |
| `AEGIS_RG` | `aegis-rg` | Resource group name. |
| `AEGIS_PG_PASSWORD` | auto-generated | Printed at the end — save it. |
| `AEGIS_ACR` | `aegisacr<digits>` | Globally-unique registry name. |

## What it creates

| Resource | SKU | Cost posture |
|---|---|---|
| Container Apps (inference, api, frontend) | Consumption | Free monthly grant; set `minReplicas: 0` in `main.bicep` for scale-to-zero. |
| PostgreSQL Flexible Server | Burstable **B1ms**, 32 GB | Free for 12 months on a new account; else ~$12–15/mo. |
| Container Registry | **Basic** | ~$5/mo — the one item with no free tier (covered by the $200 credit). |
| Log Analytics | Pay-as-you-go | First 5 GB/month free. |

On the **free account** everything above runs on the $200 / 30-day credit with **spending
protection** (your card isn't charged). To run cheaply *beyond* the credit: Postgres B1ms stays
free for 12 months, Container Apps scale-to-zero ≈ $0 when idle, and you can drop the registry by
hosting images on GHCR instead. Or just tear down between demos.

## Tear down (stop all billing)

```bash
./infra/azure/teardown.sh      # deletes the whole resource group
```

## Notes

- **Architecture:** `frontend` (public) → `api` (public, CORS-open read-only demo) → `inference`
  (internal) + Postgres. The frontend learns the API URL at runtime (`AEGIS_API_URL` → `env.js`);
  the inference image bakes the model + checkpoint + graph, so it needs no mounts in the cloud.
- **Cost control:** for a "wake on demand, ~$0 idle" demo, set `minReplicas: 0` (param in
  `main.bicep` / `-p minReplicas=0`). Trade-off: a cold start of ~20–40 s on the first hit.
- This IaC is authored to match the local stack; on a first deploy, watch the `az deployment`
  output for any region/quota/API-version messages and adjust `AEGIS_LOCATION` if a SKU isn't
  available in your region.
