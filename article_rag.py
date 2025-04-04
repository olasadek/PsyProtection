import os
import openai
import requests
import xml.etree.ElementTree as ET
from fastapi import FastAPI
from pydantic import BaseModel
from Bio import Entrez
from sentence_transformers import SentenceTransformer, util
from bs4 import BeautifulSoup
from pyngrok import ngrok
import uvicorn
import nest_asyncio
import json
import asyncio

Entrez.email = "YOUREMAIL@gmail.com"
openai.api_key = "YOUR_OPENAI_API_KEY"
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
nest_asyncio.apply()
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

def format_today():
    from datetime import datetime
    d = datetime.now()
    return f"{d.year}{d.month:02}{d.day:02}{d.hour:02}{d.minute:02}", f"{d.year-2}{d.month:02}{d.day:02}{d.hour:02}{d.minute:02}"

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
            fetch_handle = Entrez.efetch(db="pmc", id=pmcid, rettype="full", retmode="xml")
            data = fetch_handle.read()
            with open(filename, "wb") as f:
                f.write(data)
            fetch_handle.close()

def parse_pmc_articles(folder="pmc_articles"):
    articles = []
    for file in os.listdir(folder):
        if file.endswith(".xml"):
            path = os.path.join(folder, file)
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
    return articles

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

async def fetch_all(query: str) -> List[Dict]:
    """Parallel fetch with guaranteed list return"""
    pmc = await asyncio.to_thread(lambda: (
        download_pmc_articles(query),
        parse_pmc_articles()
    ))
    _, pmc_articles = pmc

    arxiv_articles = await asyncio.to_thread(fetch_arxiv_articles, query)
    return pmc_articles + arxiv_articles

def rank_articles(query, articles):
    query_embedding = sbert_model.encode(query, convert_to_tensor=True)
    article_embeddings = [sbert_model.encode(article["body"], convert_to_tensor=True) for article in articles]
    scores = [util.pytorch_cos_sim(query_embedding, emb)[0].item() for emb in article_embeddings]
    ranked_articles = sorted(zip(articles, scores), key=lambda x: x[1], reverse=True)
    return [a for a, s in ranked_articles[0:2]]

def augment_question(original_question, articles):
    context = "\n\n".join([f"Title: {a['title']}\nAbstract: {a['abstract']}" for a in articles])
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a research assistant."},
                  {"role": "user", "content": f"Given the following research articles:\n\n{context}\nRefine and augment the question: \"{original_question}\" to make it more precise."}]
    )
    return response["choices"][0]["message"]["content"]

def generate_answer(augmented_question, articles):
    context = "\n\n".join([f"Title: {a['title']}\nBody: {a['body']}" for a in articles])
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a research assistant."},
                  {"role": "user", "content": f"Given the following research articles:\n\n{context}\nAnswer the question: \"{augmented_question}\" using the most relevant information from the articles."}]
    )
    return response["choices"][0]["message"]["content"]

@app.post("/ask_question")
async def ask_question(query_request: QueryRequest):
    query = query_request.query
    all_articles = await fetch_all(query)

    # Continue with existing logic
    ranked_articles = rank_articles(query, all_articles)

    ranked_articles_path = "/content/ranked_articles.json"
    with open(ranked_articles_path, "w") as f:
        json.dump(ranked_articles, f)

    with open(ranked_articles_path, "r") as f:
        loaded_articles = json.load(f)

    augmented_question = augment_question(query, loaded_articles)
    answer = generate_answer(augmented_question, loaded_articles)

    return {"augmented_question": augmented_question, "answer": answer}

tunnel = ngrok.connect(8000)
public_url = tunnel.public_url
print(f"Public FastAPI URL: {public_url}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)