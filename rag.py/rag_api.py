from flask import Flask, request, jsonify
from utils.fetch import fetch_all
from utils.rank import rank_articles
from utils.augment import augment_question
from utils.answer import generate_answer
import json
import asyncio
from tinydb import TinyDB, Query as TinyQuery
db = TinyDB("query_cache.json")


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
