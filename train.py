import os
import torch
from tokenizers import ByteLevelBPETokenizer

from model.config import Config
from model.transformer import GPT


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

CHECKPOINT_PATH = "checkpoints/model.pt"


# -----------------------------
# Load tokenizer
# -----------------------------
tokenizer = ByteLevelBPETokenizer(
    "tokenizer/vocab.json",
    "tokenizer/merges.txt"
)


# -----------------------------
# Load dataset
# -----------------------------
with open("data/corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Characters:", len(text))

encoded = tokenizer.encode(text)
tokens = encoded.ids

data = torch.tensor(tokens, dtype=torch.long)

print("Tokens:", len(data))


# -----------------------------
# Split dataset
# -----------------------------
n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]


# -----------------------------
# Batch generator
# -----------------------------
def get_batch(split, batch_size=16):
    source = train_data if split == "train" else val_data

    if len(source) <= Config.block_size:
        raise ValueError(
            f"Dataset too small. "
            f"Tokens={len(source)}, "
            f"block_size={Config.block_size}"
        )

    ix = torch.randint(
        0,
        len(source) - Config.block_size,
        (batch_size,)
    )

    x = torch.stack([
        source[i:i + Config.block_size]
        for i in ix
    ])

    y = torch.stack([
        source[i + 1:i + Config.block_size + 1]
        for i in ix
    ])

    return x.to(device), y.to(device)


# -----------------------------
# Validation loss
# -----------------------------
@torch.no_grad()
def estimate_loss(model, eval_iters=20):
    model.eval()

    losses = {}

    for split in ["train", "val"]:
        split_losses = []

        for _ in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            split_losses.append(loss.item())

        losses[split] = sum(split_losses) / len(split_losses)

    model.train()
    return losses


# -----------------------------
# Build model
# -----------------------------
model = GPT(Config()).to(device)

print("Parameters:", sum(p.numel() for p in model.parameters()))


# -----------------------------
# Optimizer
# -----------------------------
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4
)


# -----------------------------
# Mixed precision scaler
# -----------------------------
if device == "cuda":
    scaler = torch.amp.GradScaler("cuda")
else:
    scaler = None


# -----------------------------
# Load checkpoint if compatible
# -----------------------------
if os.path.exists(CHECKPOINT_PATH):
    try:
        print("Loading checkpoint...")

        model.load_state_dict(
            torch.load(
                CHECKPOINT_PATH,
                map_location=device,
                weights_only=True
            )
        )

        print("Checkpoint loaded successfully")

    except RuntimeError:
        print("Checkpoint incompatible with current model config.")
        print("Starting fresh training.")


# -----------------------------
# Training loop
# -----------------------------
max_steps = 5000
best_val = float("inf")

for step in range(max_steps):
    xb, yb = get_batch("train")

    optimizer.zero_grad()

    if device == "cuda":
        with torch.amp.autocast("cuda"):
            _, loss = model(xb, yb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    else:
        _, loss = model(xb, yb)
        loss.backward()
        optimizer.step()

    if step % 100 == 0:
        losses = estimate_loss(model)

        print(
            f"Step {step} | "
            f"Train Loss: {losses['train']:.4f} | "
            f"Val Loss: {losses['val']:.4f}"
        )

        if losses["val"] < best_val:
            best_val = losses["val"]

            torch.save(
                model.state_dict(),
                CHECKPOINT_PATH
            )

            print("Best checkpoint saved")


print("Training complete")