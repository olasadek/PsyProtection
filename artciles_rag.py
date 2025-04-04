import os
import openai
import requests
import xml.etree.ElementTree as ET
from fastapi import FastAPI
from pydantic import BaseModel
from Bio import Entrez
from sentence_transformers import SentenceTransformer, util
from bs4 import BeautifulSoup
from neo4j import GraphDatabase
from pyngrok import ngrok
import uvicorn
import nest_asyncio

# Setup
Entrez.email = "olajsadek@gmail.com"
openai.api_key = "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"

sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
nest_asyncio.apply()
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

def format_today():
    from datetime import datetime
    d = datetime.now()
    return f"{d.year}{d.month:02}{d.day:02}{d.hour:02}{d.minute:02}", f"{d.year-2}{d.month:02}{d.day:02}{d.hour:02}{d.minute:02}"

class GraphDatabaseManager:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_article_node(self, pmid, title, abstract):
        with self.driver.session() as session:
            session.run(
                "MERGE (a:Article {pmid: $pmid, title: $title, abstract: $abstract})",
                pmid=pmid, title=title, abstract=abstract
            )

db_manager = GraphDatabaseManager(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

# Step 1: Full-text download from PMC

def download_pmc_articles(query, count=50, folder="pmc_articles"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    search_handle = Entrez.esearch(db="pmc", term=query + " AND open access[filter]", retmax=count)
    record = Entrez.read(search_handle)
    search_handle.close()
    pmc_ids = record["IdList"]

    for pmcid in pmc_ids:
        filename = os.path.join(folder, f"{pmcid}.xml")
        if not os.path.exists(filename):
            try:
                fetch_handle = Entrez.efetch(db="pmc", id=pmcid, rettype="full", retmode="xml")
                data = fetch_handle.read()
                with open(filename, "wb") as f:
                    f.write(data)
                fetch_handle.close()
            except Exception as e:
                print(f"Failed to fetch PMC {pmcid}: {e}")

# Step 2: Parse PMC XMLs

def parse_pmc_articles(folder="pmc_articles"):
    articles = []
    for file in os.listdir(folder):
        if file.endswith(".xml"):
            path = os.path.join(folder, file)
            try:
                tree = ET.parse(path)
                root = tree.getroot()
                title = root.findtext(".//article-title") or "No title"
                abstract = root.findtext(".//abstract") or "No abstract"
                body = " ".join([elem.text for elem in root.findall(".//body//p") if elem.text]) or ""
                articles.append({
                    "title": title,
                    "abstract": abstract,
                    "body": body
                })
            except Exception as e:
                print(f"Failed to parse {file}: {e}")
    return articles

# Step 3: ArXiv full-text scraping (fallback to abstract)

def fetch_arxiv_articles(search_query, max_results=5):
    today, two_years_ago = format_today()
    query = search_query.replace(" ", "+")
    url = f'http://export.arxiv.org/api/query?search_query=all:{query}&submittedDate:[{two_years_ago}+TO+{today}]&start=0&max_results={max_results}'
    response = requests.get(url)
    articles = []
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "xml")
        for entry in soup.find_all("entry"):
            title = entry.title.text
            summary = entry.summary.text
            link = entry.id.text
            articles.append({"title": title, "abstract": summary, "body": summary, "link": link})
    return articles

# Step 4: Ranking and LLM logic

def rank_articles(query, articles):
    query_embedding = sbert_model.encode(query, convert_to_tensor=True)
    article_embeddings = [sbert_model.encode(article["body"], convert_to_tensor=True) for article in articles]
    scores = [util.pytorch_cos_sim(query_embedding, emb)[0].item() for emb in article_embeddings]
    ranked_articles = sorted(zip(articles, scores), key=lambda x: x[1], reverse=True)
    return [a for a, s in ranked_articles[:5]]

def augment_question(original_question, articles):
    context = "\n\n".join([f"Title: {a['title']}\nAbstract: {a['abstract']}" for a in articles])
    prompt = f"""
    Given the following research articles:\n\n{context}
    Refine and augment the question: \"{original_question}\" to make it more precise.
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a research assistant."},
                  {"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

def generate_answer(augmented_question, articles):
    context = "\n\n".join([f"Title: {a['title']}\nBody: {a['body']}" for a in articles])
    prompt = f"""
    Given the following research articles:\n\n{context}
    Answer the question: \"{augmented_question}\" using the most relevant information from the articles.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a research assistant."},
                  {"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

# FastAPI Endpoint
@app.post("/ask_question")
async def ask_question(query_request: QueryRequest):
    query = query_request.query

    # Download if needed
    if not os.path.exists("pmc_articles") or not os.listdir("pmc_articles"):
        download_pmc_articles(query, count=50)

    pmc_articles = parse_pmc_articles()
    arxiv_articles = fetch_arxiv_articles(query)
    all_articles = pmc_articles + arxiv_articles

    ranked_articles = rank_articles(query, all_articles)
    augmented_question = augment_question(query, ranked_articles)
    answer = generate_answer(augmented_question, ranked_articles)
    return {"augmented_question": augmented_question, "answer": answer}

# Expose endpoint via ngrok
tunnel = ngrok.connect(8000)
public_url = tunnel.public_url
print(f"Public FastAPI URL: {public_url}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
