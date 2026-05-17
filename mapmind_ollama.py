import pickle
import faiss
import re
from sentence_transformers import SentenceTransformer
import ollama

INDEX_FILE = "data/vector.index"
META_FILE = "data/chunks.pkl"

STOPWORDS = {
    "what", "is", "the", "a", "an", "how",
    "explain", "tell", "me", "about",
    "can", "you", "please"
}

DOMAIN_HINTS = {
    "edi": ["edi", "850", "810", "856", "997", "as2", "x12", "edifact"],
    "oracle": ["soa", "oic", "osb", "bpel", "mediator", "oracle"],
    "xslt": ["xslt", "xpath", "xml", "xsd", "namespace"],
    "seeburger": ["seeburger", "bis"]
}

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index(INDEX_FILE)

with open(META_FILE, "rb") as f:
    chunks = pickle.load(f)


def extract_keywords(query):
    words = re.findall(r"\w+", query.lower())
    return [w for w in words if w not in STOPWORDS]


def detect_domain(query):
    q = query.lower()

    for domain, terms in DOMAIN_HINTS.items():
        for term in terms:
            if term in q:
                return domain

    return None


def keyword_score(query, text):
    keywords = extract_keywords(query)
    text_words = set(re.findall(r"\w+", text.lower()))

    score = 0

    for word in keywords:
        if word in text_words:
            score += 1

    return score


def retrieve(query, top_k=3):
    domain = detect_domain(query)

    q_embedding = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(q_embedding, 20)

    ranked = []

    for vector_score, idx in zip(scores[0], indices[0]):
        item = chunks[idx]

        kw = keyword_score(query, item["text"])

        bonus = 0

        if domain and item["category"] == domain:
            bonus = 3

        hybrid = vector_score + kw + bonus

        ranked.append((hybrid, item))

    ranked.sort(reverse=True, key=lambda x: x[0])

    return [item for _, item in ranked[:top_k]]


def ask_llm(query, context):
    context_text = "\n\n".join(
        [f"[{c['source']}]\n{c['text']}" for c in context]
    )

    prompt = f"""
You are MapMindGPT, an expert assistant for:
- EDI
- Oracle SOA / OIC / OSB
- SEEBURGER
- XML / XPath / XSLT
- AI engineering

Use ONLY the provided context if relevant.
If context is insufficient, say so.

Context:
{context_text}

Question:
{query}
"""

    response = ollama.chat(
        model="qwen2.5:7b-instruct",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]


while True:
    query = input("\nYou: ")

    if query.lower() in ["exit", "quit"]:
        break

    context = retrieve(query)

    answer = ask_llm(query, context)

    print("\nMapMindGPT:\n")
    print(answer)