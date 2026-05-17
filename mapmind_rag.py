import pickle
import faiss
import re
import torch
from tokenizers import ByteLevelBPETokenizer
from sentence_transformers import SentenceTransformer

from model.config import Config
from model.transformer import GPT

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

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- tokenizer ----------
tokenizer = ByteLevelBPETokenizer(
    "tokenizer/vocab.json",
    "tokenizer/merges.txt"
)

# ---------- GPT ----------
gpt = GPT(Config()).to(device)

gpt.load_state_dict(
    torch.load(
        "checkpoints/model.pt",
        map_location=device,
        weights_only=True
    )
)

gpt.eval()

# ---------- embeddings ----------
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


def generate_answer(query, context):
    context_text = "\n".join([c["text"] for c in context])

    prompt = (
        f"Context:\n{context_text}\n\n"
        f"User: {query}\n"
        f"Assistant:"
    )

    ids = tokenizer.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long).to(device)

    with torch.no_grad():
        out = gpt.generate(
            x,
            max_new_tokens=120,
            temperature=0.7,
            top_k=30
        )

    generated = tokenizer.decode(out[0].tolist())

    reply = generated[len(prompt):]

    if "User:" in reply:
        reply = reply.split("User:")[0]

    return reply.strip()


while True:
    query = input("\nYou: ")

    if query.lower() in ["exit", "quit"]:
        break

    context = retrieve(query)

    answer = generate_answer(query, context)

    print("\nMapMindGPT:", answer)