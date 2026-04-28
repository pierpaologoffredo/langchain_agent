from openai import OpenAI
from azure.identity import AzureCliCredential, get_bearer_token_provider

endpoint = "https://msfoundry-pgm-sde.services.ai.azure.com/api/projects/proj-default-sde/openai/v1"
deployment_name = "gpt-5.4-mini-1"
token_provider = get_bearer_token_provider(AzureCliCredential(), "https://ai.azure.com/.default")

client = OpenAI(
    base_url=endpoint,
    api_key=token_provider
)

while True:
    user_input = input("You: ")
    if user_input.lower() in ("exit", "quit"):
        break
    response = client.responses.create(
        model=deployment_name,
        input=user_input,
    )
    print(f"AI: {response.output_text}")