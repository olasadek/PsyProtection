🧠 PsyProtection RAG Research Retrieval System

This is an intelligent research retrieval system tailored for psychiatric and drug abuse treatment support. It automatically searches PubMed Central and ArXiv, ranks articles with SBERT embeddings, augments clinician queries with OpenAI, and returns treatment recommendations—caching answers for repeated use.

## 🚀 Features

- 🔍 **Dual Source Search (PMC + ArXiv)**
- 🧠 **Query Embedding & Ranking via sentence-transformers**
- 🤖 **GPT-3.5-powered Query Augmentation & Answering**
- 💾 **Caching Layer using TinyDB for repeated queries**
- 🌐 **Flask API for local or containerized deployment**

## 📦 Requirements
your open api key
your entrez email

Install dependencies:

```bash
pip install -r requirements.txt
```

Essential libraries:

```text
flask
openai
requests
biopython
sentence-transformers
beautifulsoup4
tinydb
```

## 🧪 API Endpoint

### POST /ask_question

Submit a research query (e.g., "best intervention for opioid addiction in schizophrenia").

### Request

```json
{
  "query": "What is the most effective treatment for opioid abuse in adolescents with bipolar disorder?"
}
```

### Response

```json
{
  "answer": "1. Recommended treatments: ...\n2. Alternative options: ...\n3. Special considerations: ..."
}
```

## 🧬 Core Workflow

1. 🧲 **Fetching & Parsing Research Articles**

```python
def fetch_all(query):
    pmc = await asyncio.to_thread(lambda: (
        download_pmc_articles(query),
        parse_pmc_articles()
    ))
    arxiv_articles = await asyncio.to_thread(fetch_arxiv_articles, query)
    return pmc[1] + arxiv_articles
```

2. 📊 **Semantic Ranking of Articles**

```python
def rank_articles(query, articles):
    query_embedding = sbert_model.encode(query, convert_to_tensor=True)
    article_embeddings = [sbert_model.encode(a["body"], convert_to_tensor=True) for a in articles]
    scores = [util.pytorch_cos_sim(query_embedding, emb)[0].item() for emb in article_embeddings]
    ranked = sorted(zip(articles, scores), key=lambda x: x[1], reverse=True)
    return [a for a, _ in ranked[:2]]
```

3. ✍️ **GPT-3.5 Query Augmentation**

```python
def augment_question(original_question, articles):
    context = "\n\n".join([f"Title: {a['title']}\nAbstract: {a['abstract']}" for a in articles])
    ...
```

4. 📜 **GPT-3.5 Answer Generation**

```python
def generate_answer(augmented_question, articles):
    context = "\n\n".join([f"Title: {a['title']}\nBody: {a['body']}" for a in articles])
    ...
```

5. 💾 **Caching with TinyDB**

```python
QueryEntry = TinyQuery()
cached_entry = db.get(QueryEntry.query == query)
if cached_entry:
    return jsonify({"answer": cached_entry["answer"]})
```

## 🛠️ Run the App

```bash
python app.py
```

or in Docker:

```bash
docker-compose up
```

## 🔐 Environment Setup

You should replace the hardcoded `openai.api_key` with an environment variable for production:

```python
openai.api_key = os.getenv("OPENAI_API_KEY")
```

## 📁 File Structure

```plaintext
├── augmentation.py
├── config.py
├── fetch.py
├── rag_api.py
├── augmentation.py
├── ranking.py
├── requirements.txt
```
