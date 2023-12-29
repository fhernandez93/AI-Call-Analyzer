import os
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()

keyVaultName = os.environ["KEY_VAULT_NAME"]
KVUri = f"https://{keyVaultName}.vault.azure.net"
credential = DefaultAzureCredential()
client = SecretClient(vault_url=KVUri, credential=credential)

#Retrieving credentials
HASH_KEY = str(client.get_secret("HASH-KEY").value)
TWOFA_KEY = str(client.get_secret("2fa-updated").value)
KEY_ASSEMBLYAI = str(client.get_secret("KEY-ASSEMBLYAI").value)
SQLUSER=str(client.get_secret("SQLUSER").value)
SQLPASS=str(client.get_secret("SQLPASS").value)


