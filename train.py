import argparse
import os
import torch
import math
import time
from tokenizers import ByteLevelBPETokenizer
from model.config import Config
from model.checkpoint import load_model_checkpoint, save_model_checkpoint
from model.transformer import GPT

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}")

CHECKPOINT_PATH = "checkpoints/model.pt"
FEEDBACK_TRAIN_FILE = "data/feedback_instruction_corpus.txt"
os.makedirs("checkpoints", exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MapMindGPT-Custom.")
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--preset", choices=["quick", "balanced", "full"], default="quick")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-iters", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--fresh", action="store_true", help="Start from random weights instead of loading checkpoints/model.pt.")
    parser.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    args = parser.parse_args()

    presets = {
        "quick": {"max_steps": 300, "batch_size": 8, "eval_iters": 5, "eval_interval": 50},
        "balanced": {"max_steps": 1200, "batch_size": 16, "eval_iters": 10, "eval_interval": 100},
        "full": {"max_steps": Config.max_steps, "batch_size": Config.batch_size, "eval_iters": 20, "eval_interval": 100},
    }
    selected = presets[args.preset]
    args.max_steps = args.max_steps or selected["max_steps"]
    args.batch_size = args.batch_size or selected["batch_size"]
    args.eval_iters = args.eval_iters or selected["eval_iters"]
    args.eval_interval = args.eval_interval or selected["eval_interval"]
    return args


args = parse_args()
TRAIN_FILE = args.data_file or ("data/instruction_corpus.txt" if os.path.exists("data/instruction_corpus.txt") else "data/corpus.txt")

# Load tokenizer
tokenizer = ByteLevelBPETokenizer("tokenizer/vocab.json", "tokenizer/merges.txt")

# Load dataset
print(f"Training data: {TRAIN_FILE}")
print(
    f"Preset: {args.preset} | Steps: {args.max_steps} | "
    f"Batch: {args.batch_size} | Eval iters: {args.eval_iters} | Eval every: {args.eval_interval}"
)
with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    text = f.read()

if args.data_file is None and os.path.exists(FEEDBACK_TRAIN_FILE):
    with open(FEEDBACK_TRAIN_FILE, "r", encoding="utf-8") as f:
        feedback_text = f.read().strip()
    if feedback_text:
        text = f"{text}\n\n{feedback_text}"
        print(f"Included feedback training data: {FEEDBACK_TRAIN_FILE}")

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


def get_batch(split, batch_size=None):
    batch_size = batch_size or args.batch_size
    source = train_data if split == "train" else val_data
    ix = torch.randint(0, len(source) - Config.block_size, (batch_size,))
    x = torch.stack([source[i:i + Config.block_size] for i in ix])
    y = torch.stack([source[i+1:i + Config.block_size + 1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, eval_iters=None):
    eval_iters = eval_iters or args.eval_iters
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
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

if args.fresh:
    print("Fresh training: not loading existing checkpoint")
elif os.path.exists(CHECKPOINT_PATH):
    try:
        load_model_checkpoint(model, CHECKPOINT_PATH, device)
        print("Loaded existing checkpoint")
    except RuntimeError as e:
        print(f"Checkpoint incompatible, starting fresh: {e}")

# LR Scheduler Config
max_steps = args.max_steps
warmup_steps = Config.warmup_steps
warmup_steps = min(warmup_steps, max(10, max_steps // 10))
min_lr = Config.min_learning_rate

def get_lr(it):
    if it < warmup_steps:
        return args.learning_rate * max(it + 1, 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (args.learning_rate - min_lr)

# Training Loop
best_val = float("inf")
start_time = time.time()


def format_duration(seconds):
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


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

    if step % 10 == 0 or step == max_steps - 1:
        elapsed = time.time() - start_time
        done = step + 1
        progress = done / max_steps
        eta = elapsed / progress - elapsed if progress > 0 else 0
        print(
            f"Progress {done}/{max_steps} ({progress:.1%}) | "
            f"Loss {loss.item():.4f} | Elapsed {format_duration(elapsed)} | ETA {format_duration(eta)}",
            flush=True
        )

    if step % args.eval_interval == 0 or step == max_steps - 1:
        losses = estimate_loss(model)
        print(f"Eval step {step}: Train {losses['train']:.4f}, Val {losses['val']:.4f}, LR {lr:.2e}", flush=True)
        if losses["val"] < best_val:
            best_val = losses["val"]
            save_model_checkpoint(
                model,
                CHECKPOINT_PATH,
                step=step,
                best_val=best_val,
                train_file=TRAIN_FILE,
                config={k: v for k, v in Config.__dict__.items() if not k.startswith("_")}
            )
            print(f"Best checkpoint saved at step {step} with val loss {best_val:.4f}")

print("Training complete")
