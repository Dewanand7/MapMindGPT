import os

DOCS_DIR = "docs"
OUTPUT_FILE = "data/corpus.txt"

all_text = []

for root, _, files in os.walk(DOCS_DIR):
    for file in files:
        if file.endswith(".txt"):
            path = os.path.join(root, file)

            print("Reading:", path)

            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

                all_text.append(text)

combined = "\n\n".join(all_text)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(combined)

print("Corpus built successfully.")
print("Characters:", len(combined))