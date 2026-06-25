# Activating Entra ID (Azure AD) authentication

By default AEGIS runs as a **public, read-only demo** (no auth). Authentication is **opt-in**: set the
two env vars below and the BFF turns into an OAuth2 resource server — the read endpoints stay public,
but the analyst / case-management endpoints (`/api/cases/**`, `/api/me`, writes) require a valid Entra
access token carrying an app role.

This is safe to leave off: with `AEGIS_OIDC_ISSUER` empty the security chain permits everything, so
the demo can never be locked out by accident.

## 1. Register the API (the Spring BFF)

```bash
TENANT=$(az account show --query tenantId -o tsv)

# App registration for the API (single-tenant)
API_APP=$(az ad app create --display-name "AEGIS API" --sign-in-audience AzureADMyOrg \
  --query appId -o tsv)
echo "API audience (client id): $API_APP"
echo "Issuer: https://login.microsoftonline.com/$TENANT/v2.0"
```

## 2. Define app roles (Analyst / Reviewer / Admin)

Add three app roles so tokens carry a `roles` claim (the BFF maps each to `ROLE_<name>`):

```bash
az ad app update --id "$API_APP" --app-roles '[
  {"allowedMemberTypes":["User"],"displayName":"Analyst","value":"ANALYST",
   "description":"Work cases","id":"11111111-1111-1111-1111-111111111111","isEnabled":true},
  {"allowedMemberTypes":["User"],"displayName":"Reviewer","value":"REVIEWER",
   "description":"Review/approve dispositions","id":"22222222-2222-2222-2222-222222222222","isEnabled":true},
  {"allowedMemberTypes":["User"],"displayName":"Admin","value":"ADMIN",
   "description":"Administer the system","id":"33333333-3333-3333-3333-333333333333","isEnabled":true}
]'
```

Then assign roles to users under **Entra ID → Enterprise applications → AEGIS API → Users and groups**.

## 3. Point the deployment at Entra

Redeploy with the params (or set the env directly on the running app):

```bash
# via Bicep
az deployment group create -g aegis-rg -f infra/azure/main.bicep \
  -p oidcIssuer="https://login.microsoftonline.com/$TENANT/v2.0" oidcAudience="$API_APP" \
     pgAdminPassword=... inferenceImage=... apiImage=... frontendImage=...

# or directly on the running API
az containerapp update -n aegis-api -g aegis-rg \
  --set-env-vars "AEGIS_OIDC_ISSUER=https://login.microsoftonline.com/$TENANT/v2.0" \
                 "AEGIS_OIDC_AUDIENCE=$API_APP"
```

Verify: `GET /api/flags/1` still returns 200 (public); `GET /api/me` returns **401** without a token.

## 4. Front-end login (added with the case-management UI, Phase 4)

The Angular app will use **MSAL** to sign in against a **SPA app registration** (redirect URI = the
frontend URL) and request an access token for the API's scope, which it sends as a Bearer token. Those
steps are documented alongside the workflow UI when that lands.
