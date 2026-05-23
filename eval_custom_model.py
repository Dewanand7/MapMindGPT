import json
import re

import torch
from tokenizers import ByteLevelBPETokenizer

from model.checkpoint import load_model_checkpoint
from model.config import Config
from model.transformer import GPT


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
            max_new_tokens=100,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.12,
            eos_token_id=eos_token_id,
        )

    answer = tokenizer.decode(out[0, x.size(1):].tolist()).strip()
    for marker in ["User:", "Context:", "<eos>"]:
        if marker in answer:
            answer = answer.split(marker, 1)[0].strip()
    return answer


def main():
    tokenizer = ByteLevelBPETokenizer("tokenizer/vocab.json", "tokenizer/merges.txt")
    model = GPT(Config()).to(DEVICE)
    load_model_checkpoint(model, CHECKPOINT_PATH, DEVICE)
    model.eval()

    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    passed = 0
    for item in questions:
        answer = generate_answer(model, tokenizer, item["question"])
        normalized = normalize(answer)
        hits = [kw for kw in item["expected_keywords"] if kw in normalized]
        ok = len(hits) == len(item["expected_keywords"])
        passed += int(ok)
        status = "PASS" if ok else "FAIL"
        print(f"\n[{status}] {item['question']}")
        print(f"Answer: {answer}")
        print(f"Keywords hit: {len(hits)}/{len(item['expected_keywords'])}")

    print(f"\nScore: {passed}/{len(questions)}")


if __name__ == "__main__":
    main()
