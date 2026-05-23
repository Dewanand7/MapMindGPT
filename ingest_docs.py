import os
from document_loader import is_supported_document, read_document

DOCS_DIR = "docs"
OUTPUT_FILE = "data/corpus.txt"

all_text = []

for root, _, files in os.walk(DOCS_DIR):
    for file in files:
        if is_supported_document(file):
            path = os.path.join(root, file)

            print("Reading:", path)

            text = read_document(path)

            if text.strip():
                all_text.append(text)

combined = "\n\n".join(all_text)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(combined)

print("Corpus built successfully.")
print("Characters:", len(combined))
