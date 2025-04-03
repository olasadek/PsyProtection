import os
import openai
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from Bio import Entrez
from sentence_transformers import SentenceTransformer, util
from bs4 import BeautifulSoup
from neo4j import GraphDatabase
from pyngrok import ngrok
import uvicorn
import xml.etree.ElementTree as ET
import nest_asyncio

# Secure API keys
Entrez.email = os.getenv("ENTREZ_EMAIL", "your-email@example.com")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize sentence transformer and Neo4j
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

nest_asyncio.apply()

# FastAPI initialization
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

def format_today():
    from datetime import datetime
    d = datetime.now()
    return f"{d.year}{d.month:02}{d.day:02}", f"{d.year-2}{d.month:02}{d.day:02}"

# Neo4j database connection with proper closing
class GraphDatabaseManager:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        if self.driver:
            self.driver.close()

    def __del__(self):
        self.close()

    def create_article_node(self, pmid, title, abstract):
        with self.driver.session() as session:
            session.run(
                "MERGE (a:Article {pmid: $pmid, title: $title, abstract: $abstract})",
                pmid=pmid, title=title, abstract=abstract
            )

db_manager = GraphDatabaseManager(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

# Fetch and parse PubMed articles
def search_pubmed(query, max_results=10):
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()
        return record["IdList"]
    except Exception as e:
        print(f"PubMed Search Error: {e}")
        return []

def fetch_pubmed_details(pubmed_ids):
    if not pubmed_ids:
        return
    try:
        handle = Entrez.efetch(db="pubmed", id=pubmed_ids, rettype="medline", retmode="xml")
        records = handle.read()
        handle.close()
        with open("biomed_results.xml", "w") as f:
            f.write(records)  # No .decode("utf-8")
    except Exception as e:
        print(f"PubMed Fetch Error: {e}")

def fetch_pubmed_abstracts():
    try:
        tree = ET.parse("biomed_results.xml")
        root = tree.getroot()
        articles = []
        for article in root.findall('PubmedArticle'):
            title = article.find('.//ArticleTitle')
            title_text = title.text if title is not None else "No title"
            abstract = article.find('.//Abstract/AbstractText')
            abstract_text = abstract.text if abstract is not None else "No abstract"
            articles.append({"title": title_text, "abstract": abstract_text})
        return articles
    except Exception as e:
        print(f"Error parsing PubMed XML: {e}")
        return []

# Fetch arXiv articles
def fetch_arxiv_articles(search_query, max_results=3):
    today, two_years_ago = format_today()
    query = search_query.replace(" ", "+")
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "xml")
            return [{"title": entry.title.text, "abstract": entry.summary.text, "link": entry.id.text} for entry in soup.find_all("entry")]
    except requests.RequestException as e:
        print(f"arXiv API Error: {e}")
    return []

# Rank articles by relevance
def rank_articles(query, articles):
    query_embedding = sbert_model.encode(query, convert_to_tensor=True)
    article_embeddings = [
        sbert_model.encode(article["abstract"] if article["abstract"] else "No abstract available", convert_to_tensor=True)
        for article in articles
    ]
    scores = [util.pytorch_cos_sim(query_embedding, emb)[0].item() for emb in article_embeddings]
    ranked_articles = sorted(zip(articles, scores), key=lambda x: x[1], reverse=True)
    return [a for a, s in ranked_articles[:5]]

# OpenAI-based summarization
def generate_answer(augmented_question, articles):
    context = "\n\n".join([f"Title: {a['title']}\nAbstract: {a['abstract']}" for a in articles])
    prompt = f"Given the research articles below:\n\n{context}\n\nAnswer the question: {augmented_question}"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a research assistant."},
                      {"role": "user", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"]
    except openai.error.OpenAIError as e:
        return f"OpenAI API Error: {str(e)}"

@app.post("/ask_question")
async def ask_question(query_request: QueryRequest):
    query = query_request.query
    pmids = search_pubmed(query)
    fetch_pubmed_details(pmids)
    articles = fetch_pubmed_abstracts() + fetch_arxiv_articles(query)
    ranked_articles = rank_articles(query, articles)
    answer = generate_answer(query, ranked_articles)
    return {"answer": answer}

if __name__ == "__main__":
    tunnel = ngrok.connect(8000)
    print(f"Public FastAPI URL: {tunnel.public_url}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
