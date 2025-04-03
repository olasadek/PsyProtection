import openai
import requests
from fastapi import FastAPI
from pydantic import BaseModel, Field
from Bio import Entrez
from sentence_transformers import SentenceTransformer, util
from bs4 import BeautifulSoup
from neo4j import GraphDatabase
from pyngrok import ngrok
import uvicorn
import xml.etree.ElementTree as ET
import nest_asyncio

# Set up API keys and configurations
Entrez.email = "example@gmail.com" # insert your email for pubmed
openai.api_key = "sk-your api key" # insert your api key

# Initialize sentence transformer and Neo4j
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
nest_asyncio.apply()
# FastAPI initialization
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

def format_today():
    from datetime import datetime
    d = datetime.now()
    return f"{d.year}{d.month:02}{d.day:02}{d.hour:02}{d.minute:02}", f"{d.year-2}{d.month:02}{d.day:02}{d.hour:02}{d.minute:02}"

# Neo4j database connection
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

def search_pubmed(query, max_results=10):
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]

def fetch_pubmed_details(pubmed_ids):
    handle = Entrez.efetch(db="pubmed", id=pubmed_ids, rettype="medline", retmode="xml")
    records = handle.read()
    handle.close()
    recs = records.decode("utf-8")
    with open("biomed_results.xml", "w") as f:
        f.write(recs)

def fetch_pubmed_abstracts():
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

def fetch_arxiv_articles(search_query, max_results=3):
    today, two_years_ago = format_today()
    query = search_query.replace(" ", "+")
    url = f'http://export.arxiv.org/api/query?search_query=all:{query}&submittedDate:[{two_years_ago}+TO+{today}]&start=0&max_results={max_results}'
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "xml")
        articles = []
        for entry in soup.find_all("entry"):
            title = entry.title.text
            summary = entry.summary.text
            link = entry.id.text
            articles.append({"title": title, "abstract": summary, "link": link})
        return articles
    return []

def rank_articles(query, articles):
    query_embedding = sbert_model.encode(query, convert_to_tensor=True)
    article_embeddings = [sbert_model.encode(article["abstract"], convert_to_tensor=True) for article in articles]
    scores = [util.pytorch_cos_sim(query_embedding, emb)[0].item() for emb in article_embeddings]
    ranked_articles = sorted(zip(articles, scores), key=lambda x: x[1], reverse=True)
    return [a for a, s in ranked_articles[:5]]

def augment_question(original_question, articles):
    context = "\n\n".join([f"Title: {a['title']}\nAbstract: {a['abstract']}" for a in articles])
    prompt = f"""
    Given the following research articles:\n\n{context}
    Refine and augment the question: "{original_question}" to make it more precise.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a research assistant."},
                  {"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

def generate_answer(augmented_question, articles):
    context = "\n\n".join([f"Title: {a['title']}\nAbstract: {a['abstract']}" for a in articles])
    prompt = f"""
    Given the following research articles:\n\n{context}
    Answer the question: "{augmented_question}" using the most relevant information from the articles.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a research assistant."},
                  {"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

@app.post("/ask_question")
async def ask_question(query_request: QueryRequest):
    query = query_request.query
    pmids = search_pubmed(query)
    fetch_pubmed_details(pmids)
    articles = fetch_pubmed_abstracts()
    arxiv_articles = fetch_arxiv_articles(query)
    all_articles = articles + arxiv_articles
    ranked_articles = rank_articles(query, all_articles)
    augmented_question = augment_question(query, ranked_articles)
    answer = generate_answer(augmented_question, ranked_articles)
    return {"augmented_question": augmented_question, "answer": answer}


tunnel = ngrok.connect(8000)
public_url = tunnel.public_url
print(f"Public FastAPI URL: {public_url}")

# Start Uvicorn Server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
