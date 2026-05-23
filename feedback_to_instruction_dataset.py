import argparse
import ast
import json
import os


FEEDBACK_FILE = "data/feedback.jsonl"
OUTPUT_FILE = "data/feedback_instruction_corpus.txt"


def parse_feedback_line(line):
    line = line.strip()
    if not line:
        return None

    try:
        return json.loads(line)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(line)
        except (SyntaxError, ValueError):
            return None


def clean_text(text):
    return " ".join(str(text or "").split())


def format_example(query, response):
    return f"User: {clean_text(query)}\nAssistant: {clean_text(response)}\n<eos>"


def build_examples(input_file):
    if not os.path.exists(input_file):
        return []

    examples = []
    seen = set()

    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            entry = parse_feedback_line(line)
            if not entry or entry.get("rating") != "positive":
                continue

            query = clean_text(entry.get("query"))
            response = clean_text(entry.get("response"))
            if len(query) < 3 or len(response) < 20:
                continue

            key = (query.lower(), response.lower())
            if key in seen:
                continue

            seen.add(key)
            examples.append(format_example(query, response))

    return examples


def main():
    parser = argparse.ArgumentParser(description="Convert positive feedback into instruction examples.")
    parser.add_argument("--input", default=FEEDBACK_FILE)
    parser.add_argument("--output", default=OUTPUT_FILE)
    args = parser.parse_args()

    examples = build_examples(args.input)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n\n".join(examples))

    print(f"Feedback instruction dataset built: {args.output}")
    print(f"Examples: {len(examples)}")


if __name__ == "__main__":
    main()
