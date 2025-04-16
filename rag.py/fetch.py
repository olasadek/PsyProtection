# utils/fetch.py
import os
import requests
from Bio import Entrez
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from config import entrez_email

Entrez.email = entrez_email

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
