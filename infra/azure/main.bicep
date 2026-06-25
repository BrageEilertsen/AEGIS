// AEGIS on Azure — free-tier-friendly: Container Apps (consumption) + PostgreSQL Flexible Server
// (Burstable B1ms) + Log Analytics. Images are pulled from public GHCR (built by GitHub Actions),
// so there's no ACR and no registry credentials. Deploy via infra/azure/deploy.sh.

@description('Azure region')
param location string = resourceGroup().location

@description('Short name prefix for all resources')
param prefix string = 'aegis'

@description('Public image refs (built + pushed to GHCR by .github/workflows/build-images.yml)')
param inferenceImage string = 'ghcr.io/brageeilertsen/aegis-inference:latest'
param apiImage string = 'ghcr.io/brageeilertsen/aegis-api:latest'
param frontendImage string = 'ghcr.io/brageeilertsen/aegis-frontend:latest'

@description('PostgreSQL admin username')
param pgAdminLogin string = 'aegis'

@secure()
@description('PostgreSQL admin password')
param pgAdminPassword string

@description('Allowed CORS origins for the API ("*" = any; fine for this read-only demo)')
param corsOrigins string = '*'

@description('Minimum replicas per app (0 = scale-to-zero/cheapest; 1 = always-warm)')
param minReplicas int = 1

@secure()
@description('Hugging Face token with the "Inference Providers" permission. Set -> the explanation summary is written by a hosted LLM; empty -> the instant grounded template is used.')
param hfToken string = ''

@description('Hosted instruct model id for the LLM summary (used only when hfToken is set)')
param llmModel string = 'Qwen/Qwen2.5-7B-Instruct'

@description('Entra ID (Azure AD) OIDC issuer, e.g. https://login.microsoftonline.com/<tenant>/v2.0. Empty -> public demo, auth off.')
param oidcIssuer string = ''

@description('Entra ID API audience (the app registration application/client id). Empty -> audience check skipped.')
param oidcAudience string = ''

var llmOn = !empty(hfToken)
var pgName = toLower('${prefix}-pg-${uniqueString(resourceGroup().id)}')

resource law 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: '${prefix}-logs'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

resource env 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: '${prefix}-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: law.properties.customerId
        sharedKey: law.listKeys().primarySharedKey
      }
    }
  }
}

// --- PostgreSQL Flexible Server (Burstable B1ms — free-tier eligible / credit-covered) ---
resource pg 'Microsoft.DBforPostgreSQL/flexibleServers@2023-06-01-preview' = {
  name: pgName
  location: location
  sku: { name: 'Standard_B1ms', tier: 'Burstable' }
  properties: {
    version: '16'
    administratorLogin: pgAdminLogin
    administratorLoginPassword: pgAdminPassword
    storage: { storageSizeGB: 32 }
    backup: { backupRetentionDays: 7, geoRedundantBackup: 'Disabled' }
    highAvailability: { mode: 'Disabled' }
  }
}

resource pgdb 'Microsoft.DBforPostgreSQL/flexibleServers/databases@2023-06-01-preview' = {
  parent: pg
  name: 'aegis'
  properties: { charset: 'UTF8', collation: 'en_US.utf8' }
}

resource pgfw 'Microsoft.DBforPostgreSQL/flexibleServers/firewallRules@2023-06-01-preview' = {
  parent: pg
  name: 'AllowAzureServices'
  properties: { startIpAddress: '0.0.0.0', endIpAddress: '0.0.0.0' }
}

// --- Inference service (internal: only the API reaches it) ---
resource inference 'Microsoft.App/containerApps@2024-03-01' = {
  name: '${prefix}-inference'
  location: location
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: { external: false, targetPort: 8000, transport: 'auto' }
      // The HF token (if any) is held as a secret, never as a plain env value.
      secrets: llmOn ? [ { name: 'hf-token', value: hfToken } ] : []
    }
    template: {
      containers: [
        {
          name: 'inference'
          image: inferenceImage
          resources: { cpu: json('1.0'), memory: '2.0Gi' }
          // The grounded deterministic template summary is instant, free and reliable. When an HF
          // token is provided the summary is instead written by a hosted LLM (fast + reliable, no
          // local CPU inference). Without a token the LLM stays off so the template is always used.
          env: llmOn ? [
            { name: 'AEGIS_LLM_SUMMARY', value: '1' }
            { name: 'AEGIS_HF_TOKEN', secretRef: 'hf-token' }
            { name: 'AEGIS_LLM_MODEL', value: llmModel }
          ] : [ { name: 'AEGIS_LLM_SUMMARY', value: '0' } ]
        }
      ]
      scale: { minReplicas: minReplicas, maxReplicas: 1 }
    }
  }
}

// --- Spring Boot BFF (external: the browser calls it; talks to inference + Postgres) ---
resource api 'Microsoft.App/containerApps@2024-03-01' = {
  name: '${prefix}-api'
  location: location
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: { external: true, targetPort: 8080, transport: 'auto' }
      secrets: [ { name: 'db-pwd', value: pgAdminPassword } ]
    }
    template: {
      containers: [
        {
          name: 'api'
          image: apiImage
          resources: { cpu: json('0.5'), memory: '1.0Gi' }
          env: [
            { name: 'AEGIS_DB_URL', value: 'jdbc:postgresql://${pg.properties.fullyQualifiedDomainName}:5432/aegis?sslmode=require' }
            { name: 'AEGIS_DB_USER', value: pgAdminLogin }
            { name: 'AEGIS_DB_PASSWORD', secretRef: 'db-pwd' }
            { name: 'AEGIS_INFERENCE_URL', value: 'https://${inference.properties.configuration.ingress.fqdn}' }
            { name: 'AEGIS_CORS_ORIGINS', value: corsOrigins }
            // Entra ID (Azure AD) auth: empty -> public read-only demo (resource server off); set both
            // -> analyst/case endpoints require a valid Entra access token. See infra/azure/README.md.
            { name: 'AEGIS_OIDC_ISSUER', value: oidcIssuer }
            { name: 'AEGIS_OIDC_AUDIENCE', value: oidcAudience }
          ]
        }
      ]
      scale: { minReplicas: minReplicas, maxReplicas: 2 }
    }
  }
}

// --- Angular UI (external: the public demo URL; gets the API URL injected at runtime) ---
resource frontend 'Microsoft.App/containerApps@2024-03-01' = {
  name: '${prefix}-frontend'
  location: location
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: { external: true, targetPort: 80, transport: 'auto' }
    }
    template: {
      containers: [
        {
          name: 'frontend'
          image: frontendImage
          resources: { cpu: json('0.25'), memory: '0.5Gi' }
          env: [ { name: 'AEGIS_API_URL', value: 'https://${api.properties.configuration.ingress.fqdn}/api' } ]
        }
      ]
      scale: { minReplicas: minReplicas, maxReplicas: 1 }
    }
  }
}

output frontendUrl string = 'https://${frontend.properties.configuration.ingress.fqdn}'
output apiUrl string = 'https://${api.properties.configuration.ingress.fqdn}'
