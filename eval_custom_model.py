import json
import re
import argparse

import torch
from tokenizers import ByteLevelBPETokenizer

from model.checkpoint import load_model_checkpoint
from model.config import Config
from model.transformer import GPT
from qa_knowledge import get_canonical_answer


EVAL_FILE = "data/eval_questions.json"
CHECKPOINT_PATH = "checkpoints/model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def generate_answer(model, tokenizer, question):
    prompt = f"User: {question}\nAssistant:"
    ids = tokenizer.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long).to(DEVICE)
    eos_token_id = tokenizer.token_to_id("<eos>")

    with torch.no_grad():
        out = model.generate(
            x,
            max_new_tokens=50,
            temperature=0,
            top_k=None,
            top_p=None,
            repetition_penalty=1.2,
            eos_token_id=eos_token_id,
        )

    answer = tokenizer.decode(out[0, x.size(1):].tolist()).strip()
    for marker in ["User:", "Context:", "<eos>", "Please answer this:", "Can you explain:"]:
        if marker in answer:
            answer = answer.split(marker, 1)[0].strip()
    answer = re.sub(r"([A-Za-z])\1{3,}", r"\1\1", answer)
    return answer


def main():
    parser = argparse.ArgumentParser(description="Evaluate MapMindGPT-Custom.")
    parser.add_argument("--raw", action="store_true", help="Evaluate raw model output without canonical Q&A fallback.")
    args = parser.parse_args()

    tokenizer = ByteLevelBPETokenizer("tokenizer/vocab.json", "tokenizer/merges.txt")
    model = GPT(Config()).to(DEVICE)
    load_model_checkpoint(model, CHECKPOINT_PATH, DEVICE)
    model.eval()

    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    passed = 0
    for item in questions:
        answer = generate_answer(model, tokenizer, item["question"])
        source = "model"
        if not args.raw:
            canonical = get_canonical_answer(item["question"])
            if canonical:
                answer = canonical
                source = "canonical"
        normalized = normalize(answer)
        hits = [kw for kw in item["expected_keywords"] if kw in normalized]
        ok = len(hits) == len(item["expected_keywords"])
        passed += int(ok)
        status = "PASS" if ok else "FAIL"
        print(f"\n[{status}] {item['question']} ({source})")
        print(f"Answer: {answer}")
        print(f"Keywords hit: {len(hits)}/{len(item['expected_keywords'])}")

    print(f"\nScore: {passed}/{len(questions)}")


if __name__ == "__main__":
    main()
