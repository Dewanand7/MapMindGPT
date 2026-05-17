import streamlit as st
import pickle
import faiss
import re
import os
import ollama
import torch

from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from docx import Document
from lxml import etree
from tokenizers import ByteLevelBPETokenizer

from model.config import Config
from model.transformer import GPT

INDEX_FILE = "data/vector.index"
META_FILE = "data/chunks.pkl"
DOCS_ROOT = "docs"
UPLOAD_DIR = os.path.join(DOCS_ROOT, "uploads")

os.makedirs("data", exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STOPWORDS = {
    "what", "is", "the", "a", "an", "how",
    "explain", "tell", "me", "about",
    "can", "you", "please"
}

DOMAIN_HINTS = {
    "edi": ["edi", "850", "810", "856", "997", "as2", "x12", "edifact"],
    "oracle": ["soa", "oic", "osb", "bpel", "mediator", "oracle"],
    "xslt": ["xslt", "xpath", "xml", "xsd", "namespace"],
    "seeburger": ["seeburger", "bis"],
    "ai": ["ai", "transformer", "llm", "machine", "learning"]
}

AVAILABLE_MODELS = [
    "MapMindGPT-Custom",
    "qwen2.5:7b-instruct",
    "llama3:8b",
    "mistral:7b",
    "deepseek-coder:6.7b",
    "codellama:7b"
]

AVAILABLE_MODES = [
    "General Chat",
    "Code Assistant",
    "EDI Expert",
    "Oracle Expert",
    "XSLT/XPath Expert",
    "SEEBURGER Expert"
]

st.set_page_config(page_title="MapMindGPT", layout="wide")


def get_system_prompt(mode):
    prompts = {
        "General Chat": """
You are MapMindGPT.
Use context if relevant.
""",

        "Code Assistant": """
You are an expert software engineering assistant.
Generate clean production code.
""",

        "EDI Expert": """
You are an EDI integration expert.
ANSI X12, EDIFACT, AS2, mappings, acknowledgements.
""",

        "Oracle Expert": """
You are an Oracle middleware expert.
SOA, OIC, OSB, BPEL, Mediator, dehydration, fault handling.
""",

        "XSLT/XPath Expert": """
You are an XSLT/XPath expert.
Prefer XSLT 1.0 unless requested otherwise.
""",

        "SEEBURGER Expert": """
You are a SEEBURGER BIS expert.
Partner setup, mappings, routing.
"""
    }

    return prompts.get(mode, prompts["General Chat"])


@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_custom_model():
    tokenizer = ByteLevelBPETokenizer(
        "tokenizer/vocab.json",
        "tokenizer/merges.txt"
    )

    model = GPT(Config()).to(DEVICE)

    model.load_state_dict(
        torch.load(
            "checkpoints/model.pt",
            map_location=DEVICE
        )
    )

    model.eval()

    return tokenizer, model


def generate_custom_response(prompt):
    tokenizer, model = load_custom_model()

    encoded = tokenizer.encode(prompt)
    input_ids = encoded.ids

    idx = torch.tensor(
        [input_ids],
        dtype=torch.long
    ).to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            idx,
            max_new_tokens=200,
            temperature=0.8,
            top_k=50
        )

    tokens = output[0].tolist()
    return tokenizer.decode(tokens)


def read_uploaded_file(uploaded_file):
    try:
        name = uploaded_file.name.lower()

        if name.endswith(".txt"):
            return uploaded_file.read().decode("utf-8", errors="ignore")

        elif name.endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            text = ""

            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"

            return text

        elif name.endswith(".docx"):
            doc = Document(uploaded_file)
            return "\n".join([p.text for p in doc.paragraphs])

        elif name.endswith(".xml"):
            tree = etree.parse(uploaded_file)
            return etree.tostring(
                tree,
                pretty_print=True,
                encoding="unicode"
            )

    except Exception:
        return ""

    return ""


def split_text(text, chunk_size=400, overlap=100):
    result = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            result.append(chunk)

        start += chunk_size - overlap

    return result


def scan_all_documents():
    docs = []

    for root, _, files in os.walk(DOCS_ROOT):
        for file in files:
            if file.lower().endswith((".txt", ".pdf", ".docx", ".xml")):
                docs.append(os.path.join(root, file))

    return docs


def read_file_from_disk(path):
    try:
        if not os.path.exists(path):
            return ""

        lower = path.lower()

        if lower.endswith(".txt"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

        elif lower.endswith(".pdf"):
            reader = PdfReader(path)
            text = ""

            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"

            return text

        elif lower.endswith(".docx"):
            doc = Document(path)
            return "\n".join([p.text for p in doc.paragraphs])

        elif lower.endswith(".xml"):
            tree = etree.parse(path)
            return etree.tostring(
                tree,
                pretty_print=True,
                encoding="unicode"
            )

    except Exception:
        return ""

    return ""


def rebuild_index():
    embed_model = load_embed_model()
    all_chunks = []

    docs = scan_all_documents()

    for path in docs:
        text = read_file_from_disk(path)

        if not text.strip():
            continue

        rel_path = os.path.relpath(path, DOCS_ROOT)
        category = os.path.basename(os.path.dirname(path)).lower()

        doc_chunks = split_text(text)

        for chunk in doc_chunks:
            all_chunks.append({
                "source": rel_path,
                "category": category,
                "text": chunk
            })

    texts = [c["text"] for c in all_chunks]

    if texts:
        embeddings = embed_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
    else:
        index = faiss.IndexFlatIP(384)

    faiss.write_index(index, INDEX_FILE)

    with open(META_FILE, "wb") as f:
        pickle.dump(all_chunks, f)

    st.cache_resource.clear()


@st.cache_resource
def load_resources():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
        rebuild_index()

    embed_model = load_embed_model()
    index = faiss.read_index(INDEX_FILE)

    with open(META_FILE, "rb") as f:
        chunks = pickle.load(f)

    return embed_model, index, chunks


embed_model, index, chunks = load_resources()


def add_uploaded_document(uploaded_file):
    text = read_uploaded_file(uploaded_file)

    if not text:
        return False

    save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)

    rebuild_index()
    return True


def get_uploaded_documents():
    files = []

    for file in os.listdir(UPLOAD_DIR):
        full = os.path.join(UPLOAD_DIR, file)

        if os.path.isfile(full):
            files.append(file)

    return sorted(files)


def delete_uploaded_document(filename):
    path = os.path.join(UPLOAD_DIR, filename)

    if os.path.exists(path):
        os.remove(path)

    rebuild_index()


def clear_uploaded_knowledge():
    for file in os.listdir(UPLOAD_DIR):
        full = os.path.join(UPLOAD_DIR, file)

        if os.path.isfile(full):
            os.remove(full)

    rebuild_index()


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
    return sum(1 for word in keywords if word in text_words)


def retrieve(query, top_k=3):
    if not chunks:
        return []

    q_embedding = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(q_embedding, min(20, len(chunks)))

    domain = detect_domain(query)
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


def ask_llm(query, context, history, model_name, mode):
    context_text = "\n\n".join(
        [f"[{c['source']}]\n{c['text']}" for c in context]
    )

    system_prompt = get_system_prompt(mode)

    if model_name == "MapMindGPT-Custom":
        prompt = f"""
{system_prompt}

Context:
{context_text}

Question:
{query}

Answer:
"""
        return generate_custom_response(prompt)

    messages = [{
        "role": "system",
        "content": system_prompt
    }]

    messages.extend(history)

    messages.append({
        "role": "user",
        "content": f"""
Context:
{context_text}

Question:
{query}
"""
    })

    response = ollama.chat(
        model=model_name,
        messages=messages
    )

    return response["message"]["content"]


st.title("MapMindGPT")

st.sidebar.header("AI Settings")

selected_model = st.sidebar.selectbox(
    "Choose Model",
    AVAILABLE_MODELS
)

selected_mode = st.sidebar.selectbox(
    "Choose Mode",
    AVAILABLE_MODES
)

st.sidebar.header("Knowledge Base")

uploaded_file = st.sidebar.file_uploader(
    "Upload Document",
    type=["txt", "pdf", "docx", "xml"]
)

if uploaded_file:
    if add_uploaded_document(uploaded_file):
        st.sidebar.success("Document uploaded and indexed.")
        st.rerun()

uploaded_docs = get_uploaded_documents()

if uploaded_docs:
    selected_doc = st.sidebar.selectbox(
        "Uploaded Documents",
        uploaded_docs
    )

    if st.sidebar.button("Delete Selected"):
        delete_uploaded_document(selected_doc)
        st.rerun()

if st.sidebar.button("Rebuild Index"):
    rebuild_index()
    st.rerun()

if st.sidebar.button("Clear Uploaded Knowledge"):
    clear_uploaded_knowledge()
    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask MapMindGPT...")

if query:
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    context = retrieve(query)
    history = st.session_state.messages[-6:]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask_llm(
                query,
                context,
                history,
                selected_model,
                selected_mode
            )

        st.markdown(answer)

        with st.expander("Sources"):
            for c in context:
                st.markdown(f"**{c['source']}**")
                st.code(c["text"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })