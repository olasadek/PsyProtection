import openai
from config import openai_api_key

openai.api_key = openai_api_key

def augment_question(original_question, articles):
    context = "\n\n".join([f"Title: {a['title']}\nAbstract: {a['abstract']}" for a in articles])
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a research assistant."},
                  {"role": "user", "content": f"Given the following research articles:\n\n{context}\nRefine and augment the question: \"{original_question}\" to make it more precise."}]
    )
    return response["choices"][0]["message"]["content"]
