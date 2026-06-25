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
    }
    template: {
      containers: [
        {
          name: 'inference'
          image: inferenceImage
          resources: { cpu: json('1.0'), memory: '2.0Gi' }
          // The grounded deterministic template summary is instant, free and reliable. The optional
          // local LLM rephrasing (AEGIS_LLM_SUMMARY=1) needs fp32 + more memory to be reliable on
          // CPU — bf16 is unstable across the Container Apps hardware fleet — so it's off by default.
          env: [ { name: 'AEGIS_LLM_SUMMARY', value: '0' } ]
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
