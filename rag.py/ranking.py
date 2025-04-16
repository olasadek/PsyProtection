from sentence_transformers import SentenceTransformer, util
from config import sbert_model_name

sbert_model = SentenceTransformer(sbert_model_name)

def rank_articles(query, articles):
    query_embedding = sbert_model.encode(query, convert_to_tensor=True)
    article_embeddings = [sbert_model.encode(article["body"], convert_to_tensor=True) for article in articles]
    scores = [util.pytorch_cos_sim(query_embedding, emb)[0].item() for emb in article_embeddings]
    ranked_articles = sorted(zip(articles, scores), key=lambda x: x[1], reverse=True)
    return [a for a, s in ranked_articles[0:2]]
