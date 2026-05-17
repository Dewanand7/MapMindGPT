import pickle
import faiss
import re
from sentence_transformers import SentenceTransformer

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

model = SentenceTransformer("all-MiniLM-L6-v2")
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


while True:
    query = input("\nYou: ")

    if query.lower() in ["exit", "quit"]:
        break

    domain = detect_domain(query)

    q_embedding = model.encode(
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

    print("\nTop matches:\n")

    for score, item in ranked[:3]:
        print(f"Category: {item['category']}")
        print(f"Source: {item['source']}")
        print(f"Score: {score:.4f}")
        print(item["text"])
        print("=" * 80)