# Deploying AEGIS to Azure

A deploy of the full stack to **Azure Container Apps** + **Azure Database for PostgreSQL Flexible
Server**, sized for the **Azure free account** ($200 credit / free tiers, spending protection).

Images are built by **GitHub Actions** and published to **GHCR** (GitHub's container registry), then
Azure pulls them. This avoids ACR Tasks (blocked on free subscriptions) and needs **no local Docker**.

## How it works

```
push to GitHub ─▶ GitHub Actions builds 3 images ─▶ GHCR (public) ─▶ Azure Container Apps
```

## One-time setup

1. **Azure account** — the free one works: https://azure.microsoft.com/free
2. **Azure CLI** — `brew install azure-cli` (macOS) or https://aka.ms/InstallAzureCLI
3. **Build the images** — they build automatically on every push (workflow:
   `.github/workflows/build-images.yml`). Check the repo's **Actions** tab; the `build-images` run
   should finish green (~10 min). To trigger manually: Actions → build-images → *Run workflow*.
4. **Make the 3 GHCR packages public** (so Azure can pull them without credentials) — one time:
   GitHub → your profile → **Packages** → for each of `aegis-inference`, `aegis-api`,
   `aegis-frontend`: *Package settings* → *Change visibility* → **Public**.

## Deploy

```bash
az login
./infra/azure/deploy.sh        # prints the public frontend URL at the end (~3-5 min)
```

Open the printed **Frontend URL**. First hit wakes the apps and warms the model — give it ~a minute.

Customise via env vars:
```bash
AEGIS_LOCATION=westeurope ./infra/azure/deploy.sh     # region
AEGIS_GHCR_OWNER=youruser ./infra/azure/deploy.sh     # if your GH owner differs
```

## What it creates

| Resource | SKU | Cost posture |
|---|---|---|
| Container Apps (inference, api, frontend) | Consumption | Free monthly grant; `-p minReplicas=0` for scale-to-zero. |
| PostgreSQL Flexible Server | Burstable **B1ms**, 32 GB | Free 12 months on a new account; else ~$12–15/mo. |
| Log Analytics | Pay-as-you-go | First 5 GB/month free. |
| Registry | **GHCR** (GitHub) | **Free** for public images. |

On the free account this runs on the $200 / 30-day credit with **spending protection**. Beyond the
credit: Postgres B1ms is free for 12 months, Container Apps scale-to-zero ≈ $0 idle, GHCR is free.

## Tear down (stop all billing)

```bash
./infra/azure/teardown.sh      # deletes the whole resource group
```

## Notes

- **Architecture:** `frontend` (public) → `api` (public, CORS-open read-only demo) → `inference`
  (internal) + Postgres. The frontend gets the API URL at runtime (`AEGIS_API_URL` → `env.js`); the
  inference image bakes the model + checkpoint + graph, so it needs no mounts in the cloud.
- **Cheapest "wake on demand" demo:** `az deployment group create ... -p minReplicas=0` (≈ $0 idle;
  ~20–40 s cold start on first hit).
- This IaC is authored to match the local stack; on a first deploy, watch the `az deployment`
  output for any region/quota/API-version messages and adjust `AEGIS_LOCATION` if a SKU isn't
  available in your region.
