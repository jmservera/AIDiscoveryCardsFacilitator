param location string
@minLength(3)
@maxLength(22)
param resourceToken string
param tags object
param principalId string

var abbrs = loadJsonContent('./abbreviations.json')

resource azureOpenAI 'Microsoft.CognitiveServices/accounts@2024-04-01-preview' = {
  name: '${abbrs.cognitiveServicesAccounts}${resourceToken}'
  location: location
  kind: 'OpenAI'
  sku: {
    name: 'S0'
  }
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    customSubDomainName: '${abbrs.cognitiveServicesAccounts}${resourceToken}'
  }
}

var deployments = [
  {
    name: 'gpt-4o'
    skuName: 'GlobalStandard'
    modelVersion: '2024-08-06'
  }
  {
    name: 'text-embedding-ada-002'
    skuName: 'Standard'
    modelVersion: '2'
  }
  {
    name: 'o4-mini'
    skuName: 'GlobalStandard'
    modelVersion: '2025-04-16'
  }
]

@batchSize(1)
resource azureOpenAIModel 'Microsoft.CognitiveServices/accounts/deployments@2024-04-01-preview' = [
  for deployment in deployments: {
    name: deployment.name
    parent: azureOpenAI
    sku: {
      name: deployment.skuName
      capacity: 100
    }
    properties: {
      model: {
        format: 'OpenAI'
        name: deployment.name
        version: deployment.modelVersion
      }
    }
  }
]

resource web 'Microsoft.Web/sites@2022-03-01' = {
  name: 'web-${resourceToken}'
  location: location
  tags: union(tags, { 'azd-service-name': 'web' })
  kind: 'app,linux'
  properties: {
    serverFarmId: appServicePlan.id
    siteConfig: {
      linuxFxVersion: 'PYTHON|3.12'
      ftpsState: 'Disabled'
      appCommandLine: 'startup.sh'
    }
    httpsOnly: true
  }
  identity: {
    type: 'SystemAssigned'
  }

  resource appSettings 'config' = {
    name: 'appsettings'
    properties: {
      SCM_DO_BUILD_DURING_DEPLOYMENT: 'true'
      ENABLE_ORYX_BUILD: 'true'
      AZURE_OPENAI_ENDPOINT: 'https://${azureOpenAI.name}.openai.azure.com/'
    }
  }

  resource logs 'config' = {
    name: 'logs'
    properties: {
      applicationLogs: {
        fileSystem: {
          level: 'Verbose'
        }
      }
      detailedErrorMessages: {
        enabled: true
      }
      failedRequestsTracing: {
        enabled: true
      }
      httpLogs: {
        fileSystem: {
          enabled: true
          retentionInDays: 1
          retentionInMb: 35
        }
      }
    }
  }
}
resource appServicePlan 'Microsoft.Web/serverfarms@2022-03-01' = {
  name: 'app-${resourceToken}'
  location: location
  sku: {
    name: 'B1'
    tier: 'Basic'
    capacity: 1
  }
  kind: 'linux'
  properties: {
    reserved: true
  }
}

resource cognitiveServicesOpenAIUserForUser 'Microsoft.Authorization/roleAssignments@2020-04-01-preview' = {
  scope: azureOpenAI
  name: guid(
    azureOpenAI.id,
    principalId,
    resourceId('Microsoft.Authorization/roleDefinitions', '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd')
  )
  properties: {
    roleDefinitionId: resourceId('Microsoft.Authorization/roleDefinitions', '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd')
    principalId: principalId
    principalType: 'User'
  }
}

resource cognitiveServicesOpenAIUserForAppService 'Microsoft.Authorization/roleAssignments@2020-04-01-preview' = {
  scope: azureOpenAI
  name: guid(
    azureOpenAI.id,
    web.id,
    resourceId('Microsoft.Authorization/roleDefinitions', '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd')
  )
  properties: {
    roleDefinitionId: resourceId('Microsoft.Authorization/roleDefinitions', '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd')
    principalId: web.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

output WEB_URI string = 'https://${web.properties.defaultHostName}'
output AZURE_OPENAI_ENDPOINT string = 'https://${azureOpenAI.name}.openai.azure.com/'
