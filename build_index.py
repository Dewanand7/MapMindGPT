import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from document_loader import is_supported_document, read_document

DOCS_DIR = "docs"
INDEX_FILE = "data/vector.index"
META_FILE = "data/chunks.pkl"

os.makedirs("data", exist_ok=True)
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
        if is_supported_document(file):
            path = os.path.join(root, file)

            print("Reading:", path)

            text = read_document(path)
            if not text.strip():
                continue

            doc_chunks = split_text(text)

            category = os.path.basename(root).lower()

            for chunk in doc_chunks:
                chunks.append({
                    "id": len(chunks),
                    "source": os.path.relpath(path, DOCS_DIR),
                    "category": category,
                    "text": chunk
                })

texts = [c["text"] for c in chunks]

print("Total chunks:", len(texts))

dimension = model.get_sentence_embedding_dimension()
index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))

if texts:
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)
    ids = np.array([chunk["id"] for chunk in chunks], dtype=np.int64)
    index.add_with_ids(embeddings, ids)

faiss.write_index(index, INDEX_FILE)

with open(META_FILE, "wb") as f:
    pickle.dump(chunks, f)

print("Vector index built successfully.")
