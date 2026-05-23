import os
import torch
import math
from tokenizers import ByteLevelBPETokenizer
from model.config import Config
from model.checkpoint import load_model_checkpoint, save_model_checkpoint
from model.transformer import GPT

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}")

CHECKPOINT_PATH = "checkpoints/model.pt"
os.makedirs("checkpoints", exist_ok=True)

# Load tokenizer
tokenizer = ByteLevelBPETokenizer("tokenizer/vocab.json", "tokenizer/merges.txt")

# Load dataset
with open("data/corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokens = tokenizer.encode(text).ids
data = torch.tensor(tokens, dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def validate_split(name, source):
    min_tokens = Config.block_size + 1
    if len(source) < min_tokens:
        raise ValueError(
            f"{name} split is too small for training. "
            f"Need at least {min_tokens} tokens, got {len(source)}."
        )


validate_split("train", train_data)
validate_split("val", val_data)


def get_batch(split, batch_size=Config.batch_size):
    source = train_data if split == "train" else val_data
    ix = torch.randint(0, len(source) - Config.block_size, (batch_size,))
    x = torch.stack([source[i:i + Config.block_size] for i in ix])
    y = torch.stack([source[i+1:i + Config.block_size + 1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, eval_iters=20):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

model = GPT(Config()).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=0.1)
scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

if os.path.exists(CHECKPOINT_PATH):
    try:
        load_model_checkpoint(model, CHECKPOINT_PATH, device)
        print("Loaded existing checkpoint")
    except RuntimeError as e:
        print(f"Checkpoint incompatible, starting fresh: {e}")

# LR Scheduler Config
max_steps = Config.max_steps
warmup_steps = Config.warmup_steps
min_lr = Config.min_learning_rate

def get_lr(it):
    if it < warmup_steps:
        return Config.learning_rate * it / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (Config.learning_rate - min_lr)

# Training Loop
best_val = float("inf")
for step in range(max_steps):
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    xb, yb = get_batch("train")

    if device == "cuda":
        with torch.amp.autocast("cuda"):
            _, loss = model(xb, yb)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        _, loss = model(xb, yb)
        loss.backward()
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)

    if step % 100 == 0:
        losses = estimate_loss(model)
        print(f"Step {step}: Train {losses['train']:.4f}, Val {losses['val']:.4f}, LR {lr:.2e}")
        if losses["val"] < best_val:
            best_val = losses["val"]
            save_model_checkpoint(
                model,
                CHECKPOINT_PATH,
                step=step,
                best_val=best_val,
                config={k: v for k, v in Config.__dict__.items() if not k.startswith("_")}
            )
