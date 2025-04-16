# app.py
from flask import Flask, request, jsonify
from utils.fetch import fetch_all
from utils.rank import rank_articles
from utils.augment import augment_question
from utils.answer import generate_answer
import json
import asyncio

app = Flask(__name__)

@app.route('/ask_question', methods=['POST'])
def ask_question():
    data = request.get_json()
    query_request = QueryRequest(query=data.get('query'))

    query = query_request.query
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    all_articles = loop.run_until_complete(fetch_all(query))

    ranked_articles = rank_articles(query, all_articles)

    ranked_articles_path = "ranked_articles.json"
    with open(ranked_articles_path, "w") as f:
        json.dump(ranked_articles, f)

    with open(ranked_articles_path, "r") as f:
        loaded_articles = json.load(f)

    augmented_question = augment_question(query, loaded_articles)
    answer = generate_answer(augmented_question, loaded_articles)

    return jsonify({
        "answer": answer
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
