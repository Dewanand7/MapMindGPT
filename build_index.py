import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

DOCS_DIR = "docs"
INDEX_FILE = "data/vector.index"
META_FILE = "data/chunks.pkl"

model = SentenceTransformer("all-MiniLM-L6-v2")

chunks = []


def split_text(text, chunk_size=300, overlap=80):
    result = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            result.append(chunk)

        start += chunk_size - overlap

    return result


for root, _, files in os.walk(DOCS_DIR):
    for file in files:
        if file.endswith(".txt"):
            path = os.path.join(root, file)

            print("Reading:", path)

            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

            doc_chunks = split_text(text)

            category = os.path.basename(root).lower()

            for chunk in doc_chunks:
                chunks.append({
                    "source": file,
                    "category": category,
                    "text": chunk
                })

texts = [c["text"] for c in chunks]

print("Total chunks:", len(texts))

embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    normalize_embeddings=True
)

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

faiss.write_index(index, INDEX_FILE)

with open(META_FILE, "wb") as f:
    pickle.dump(chunks, f)

print("Vector index built successfully.")