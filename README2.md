# Psychiatrist assistant API

A FastAPI service that fetches and analyzes research papers from PMC and arXiv.

## Features
- Parallel article fetching
- GPT-3.5-turbo powered question augmentation
- Semantic ranking of papers

## Setup
1. Install requirements: `pip install -r requirements.txt`
2. Set environment variables:
   ```bash
   export OPENAI_API_KEY="your_key"
   export ENTREZ_EMAIL="your_email"