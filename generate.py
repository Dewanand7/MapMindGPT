import torch
from tokenizers import ByteLevelBPETokenizer

from model.config import Config
from model.checkpoint import load_model_checkpoint
from model.transformer import GPT

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = ByteLevelBPETokenizer(
    "tokenizer/vocab.json",
    "tokenizer/merges.txt"
)

model = GPT(Config()).to(device)
load_model_checkpoint(model, "checkpoints/model.pt", device)

model.eval()


def generate_reply(prompt):
    formatted = f"User: {prompt}\nAssistant:"

    ids = tokenizer.encode(formatted).ids
    x = torch.tensor([ids], dtype=torch.long).to(device)
    eos_token_id = tokenizer.token_to_id("<eos>")

    with torch.no_grad():
        out = model.generate(
            x,
            max_new_tokens=80,
            temperature=0.75,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.12,
            eos_token_id=eos_token_id
        )

    generated = tokenizer.decode(out[0, x.size(1):].tolist())

    reply = generated.strip()

    for marker in ["User:", "Assistant:", "<eos>"]:
        if marker in reply:
            reply = reply.split(marker, 1)[0]

    return reply.strip()


while True:
    prompt = input("\nYou: ")

    if prompt.lower() in ["exit", "quit"]:
        break

    try:
        answer = generate_reply(prompt)
        print("\nMapMindGPT:", answer)

    except Exception as e:
        print("Error:", e)
