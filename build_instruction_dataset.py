import argparse
import os
import random
import re
from document_loader import is_supported_document, read_document


DOCS_DIR = "docs"
OUTPUT_FILE = "data/instruction_corpus.txt"

SEED_QA = [
    (
        "What is EDI?",
        "EDI, or Electronic Data Interchange, is the structured computer-to-computer exchange of business documents between trading partners.",
    ),
    (
        "What is an EDI 850?",
        "EDI 850 is an ANSI X12 purchase order transaction sent from a buyer to a seller.",
    ),
    (
        "What is an EDI 810?",
        "EDI 810 is an ANSI X12 invoice transaction sent by a supplier to request payment.",
    ),
    (
        "What is an EDI 856?",
        "EDI 856 is an advance ship notice that communicates shipment contents and tracking details.",
    ),
    (
        "What is an EDI 997?",
        "EDI 997 is a functional acknowledgement that confirms whether an EDI transaction was accepted or rejected.",
    ),
    (
        "What is AS2?",
        "AS2 is a secure protocol commonly used to exchange EDI documents over HTTP or HTTPS.",
    ),
    (
        "What is XSLT?",
        "XSLT is a language used to transform XML documents into another XML, text, or markup format.",
    ),
    (
        "What is XPath?",
        "XPath is an expression language used to select nodes and values from XML documents.",
    ),
    (
        "What is Oracle OIC?",
        "Oracle Integration Cloud is Oracle's cloud integration platform for connecting applications, services, and data.",
    ),
    (
        "What is SEEBURGER BIS?",
        "SEEBURGER BIS is a B2B integration platform used for EDI, partner onboarding, routing, and message processing.",
    ),
]

QUESTION_TEMPLATES = [
    "Explain this {domain} topic.",
    "Summarize the key point from this {domain} note.",
    "What should I know about this {domain} concept?",
    "Give a practical answer based on this {domain} context.",
]


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_text(text, chunk_size=700, overlap=120):
    chunks = []
    start = 0
    while start < len(text):
        chunk = clean_text(text[start:start + chunk_size])
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def read_docs():
    docs = []
    for root, _, files in os.walk(DOCS_DIR):
        for file in files:
            if not is_supported_document(file):
                continue
            path = os.path.join(root, file)
            text = read_document(path)
            if not text.strip():
                continue
            domain = os.path.basename(root).lower()
            source = os.path.relpath(path, DOCS_DIR)
            docs.append((domain, source, text))
    return docs


def format_example(question, answer, context=None, source=None):
    parts = []
    if context:
        parts.append(f"Context:\n{context}")
    if source:
        parts.append(f"Source: {source}")
    parts.append(f"User: {question}")
    parts.append(f"Assistant: {answer}")
    parts.append("<eos>")
    return "\n".join(parts)


def build_examples(seed_repeats, doc_repeats):
    examples = []

    for _ in range(seed_repeats):
        for question, answer in SEED_QA:
            examples.append(format_example(question, answer))
            examples.append(format_example(f"Can you explain: {question}", answer))

    for domain, source, text in read_docs():
        for chunk in split_text(text):
            answer = chunk[:500].rstrip()
            for _ in range(doc_repeats):
                question = random.choice(QUESTION_TEMPLATES).format(domain=domain.upper())
                examples.append(format_example(question, answer, context=chunk, source=source))

    random.shuffle(examples)
    return examples


def main():
    parser = argparse.ArgumentParser(description="Build instruction-style corpus for MapMindGPT-Custom.")
    parser.add_argument("--output", default=OUTPUT_FILE)
    parser.add_argument("--seed-repeats", type=int, default=60)
    parser.add_argument("--doc-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    examples = build_examples(args.seed_repeats, args.doc_repeats)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n\n".join(examples))

    print(f"Instruction dataset built: {args.output}")
    print(f"Examples: {len(examples)}")


if __name__ == "__main__":
    main()
