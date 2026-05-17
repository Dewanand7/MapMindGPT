import torch
from tokenizers import ByteLevelBPETokenizer

from model.config import Config
from model.transformer import GPT

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = ByteLevelBPETokenizer(
    "tokenizer/vocab.json",
    "tokenizer/merges.txt"
)

model = GPT(Config()).to(device)

model.load_state_dict(
    torch.load(
        "checkpoints/model.pt",
        map_location=device,
        weights_only=True
    )
)

model.eval()


def generate_reply(prompt):
    formatted = f"User: {prompt}\nAssistant:"

    ids = tokenizer.encode(formatted).ids
    x = torch.tensor([ids], dtype=torch.long).to(device)

    with torch.no_grad():
        out = model.generate(
            x,
            max_new_tokens=40,   # reduced
            temperature=0.7,
            top_k=20             # reduced
        )

    generated = tokenizer.decode(out[0].tolist())

    reply = generated[len(formatted):]

    if "User:" in reply:
        reply = reply.split("User:")[0]

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