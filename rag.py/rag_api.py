import os
import openai
import requests
import xml.etree.ElementTree as ET
from flask import Flask, request, jsonify
from Bio import Entrez
from sentence_transformers import SentenceTransformer, util
from bs4 import BeautifulSoup
import json
import asyncio
from tinydb import TinyDB, Query as TinyQuery

Entrez.email = "YOURMAIL@gmail.com"
openai.api_key = "YOUR_API_KEY"

sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

app = Flask(__name__)

db = TinyDB("query_cache.json")

class QueryRequest:
    def __init__(self, query):
        self.query = query

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

async def fetch_all(query: str):
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
        messages=[
            {"role": "system", "content": "You are a clinical addiction specialist. Provide only direct treatment recommendations without commentary."},
            {"role": "user", "content": f"""Based on these research articles:
            {context}
            
            Answer this clinical question concisely:
            {augmented_question}
            
            Structure as:
            1. Recommended treatments
            2. Alternative options
            3. Special considerations
            """}
        ]
    )
    return response["choices"][0]["message"]["content"]

@app.route('/ask_question', methods=['POST'])
def ask_question():
    data = request.get_json()
    query = data.get("query")

    QueryEntry = TinyQuery()
    cached_entry = db.get(QueryEntry.query == query)

    if cached_entry:
        return jsonify({"answer": cached_entry["answer"]})

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    all_articles = loop.run_until_complete(fetch_all(query))
    ranked_articles = rank_articles(query, all_articles)

    augmented_question = augment_question(query, ranked_articles)
    answer = generate_answer(augmented_question, ranked_articles)

    db.insert({"query": query, "answer": answer})

    return jsonify({"answer": answer})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

app = Flask(__name__)

@app.route('/ask_question', methods=['POST'])
def ask_question():
    data = request.get_json()
    query = data.get("query")

    QueryEntry = TinyQuery()
    cached_entry = db.get(QueryEntry.query == query)

    if cached_entry:
        return jsonify({"answer": cached_entry["answer"]})

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    all_articles = loop.run_until_complete(fetch_all(query))
    ranked_articles = rank_articles(query, all_articles)

    augmented_question = augment_question(query, ranked_articles)
    answer = generate_answer(augmented_question, ranked_articles)

    db.insert({"query": query, "answer": answer})

    return jsonify({"answer": answer})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
