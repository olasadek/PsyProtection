import os

def read_secret(path):
    with open(path, "r") as f:
        return f.read().strip()

openai_api_key = read_secret("/run/secrets/openai_api_key")
entrez_email = read_secret("/run/secrets/entrez_email")
