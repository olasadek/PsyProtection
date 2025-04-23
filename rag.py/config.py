import os

openai_api_key = os.environ.get("openai_api_key")
entrez_email = os.environ.get("entrez_email")

if not openai_api_key or not entrez_email:
    raise ValueError("Missing environment variables: openai_api_key or entrez_email")
