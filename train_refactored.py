"""
Refactored train.py with improved error handling and configuration management.
"""
import argparse
import os
import torch
import math
import time
import logging
from pathlib import Path
from typing import Dict, Tuple

from tokenizers import ByteLevelBPETokenizer
from config import Config, TrainingConfig
from logging_config import LogManager, get_logger
from model.config import Config as ModelConfig
from model.checkpoint import load_model_checkpoint, save_model_checkpoint
from model.transformer import GPT


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with validation."""
    parser = argparse.ArgumentParser(
        description="Train MapMindGPT-Custom model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --preset balanced
  python train.py --preset full --max-steps 2000
  python train.py --preset quick --fresh
        """
    )
    
    parser.add_argument(
        "--data-file",
        default=None,
        help="Path to training data file (default: instruction_corpus.txt or corpus.txt)"
    )
    parser.add_argument(
        "--preset",
        choices=["quick", "balanced", "full"],
        default="quick",
        help="Training preset (quick: 300 steps, balanced: 1200, full: max)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override preset max steps"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override preset batch size"
    )
    parser.add_argument(
        "--eval-iters",
        type=int,
        default=None,
        help="Override preset eval iterations"
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="Override preset eval interval"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=ModelConfig.learning_rate,
        help="Learning rate"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start from random weights instead of loading checkpoint"
    )
    
    return parser.parse_args()


def get_training_file(data_file: str = None) -> Path:
    """
    Determine training file with proper validation.
    
    Args:
        data_file: Explicit data file path
        
    Returns:
        Path to training file
        
    Raises:
        FileNotFoundError: If no training file found
    """
    if data_file:
        path = Path(data_file)
        if not path.exists():
            raise FileNotFoundError(f"Specified data file not found: {path}")
        logger.info(f"Using explicit data file: {path}")
        return path
    
    # Try instruction corpus first, then fallback
    candidates = [
        Config.INSTRUCTION_CORPUS_FILE,
        Config.CORPUS_FILE,
    ]
    
    for candidate in candidates:
        if candidate.exists():
            logger.info(f"Using training file: {candidate}")
            return candidate
    
    raise FileNotFoundError(
        f"No training data found. Tried: {', '.join(str(c) for c in candidates)}\n"
        f"Create with: python generate_dataset.py"
    )


def load_tokenizer(vocab_file: Path, merges_file: Path) -> ByteLevelBPETokenizer:
    """
    Load tokenizer with error handling.
    
    Args:
        vocab_file: Path to vocab.json
        merges_file: Path to merges.txt
        
    Returns:
        Loaded tokenizer
        
    Raises:
        FileNotFoundError: If tokenizer files not found
    """
    if not vocab_file.exists():
        raise FileNotFoundError(f"Tokenizer vocab not found: {vocab_file}")
    if not merges_file.exists():
        raise FileNotFoundError(f"Tokenizer merges not found: {merges_file}")
    
    try:
        tokenizer = ByteLevelBPETokenizer(str(vocab_file), str(merges_file))
        logger.info(f"Loaded tokenizer from {vocab_file.parent}")
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise


def load_training_data(
    train_file: Path,
    tokenizer: ByteLevelBPETokenizer,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load and tokenize training data.
    
    Args:
        train_file: Path to training data
        tokenizer: Tokenizer instance
        device: Device to place data on
        
    Returns:
        (train_data, val_data) tensors
        
    Raises:
        ValueError: If data split is too small
    """
    logger.info(f"Loading training data from {train_file}")
    
    with open(train_file, "r", encoding="utf-8") as f:
        text = f.read()
    
    if not text.strip():
        raise ValueError(f"Training file is empty: {train_file}")
    
    logger.info(f"Tokenizing {len(text):,} characters...")
    tokens = tokenizer.encode(text).ids
    data = torch.tensor(tokens, dtype=torch.long)
    
    logger.info(f"Total tokens: {len(data):,}")
    
    # Split into train/val (90/10)
    n = int(0.9 * len(data))
    train_data = data[:n].to(device)
    val_data = data[n:].to(device)
    
    # Validate split sizes
    min_tokens = ModelConfig.block_size + 1
    if len(train_data) < min_tokens:
        raise ValueError(
            f"Training split too small ({len(train_data)} tokens). "
            f"Need at least {min_tokens} tokens."
        )
    if len(val_data) < min_tokens:
        raise ValueError(
            f"Validation split too small ({len(val_data)} tokens). "
            f"Need at least {min_tokens} tokens."
        )
    
    logger.info(f"Train: {len(train_data):,} tokens | Val: {len(val_data):,} tokens")
    return train_data, val_data


def get_batch(
    split: str,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get a batch of training data."""
    source = train_data if split == "train" else val_data
    ix = torch.randint(0, len(source) - ModelConfig.block_size, (batch_size,))
    x = torch.stack([source[i:i + ModelConfig.block_size] for i in ix])
    y = torch.stack([source[i+1:i + ModelConfig.block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    eval_iters: int,
    device: str
) -> Dict[str, float]:
    """Estimate loss on train and validation sets."""
    model.eval()
    losses = {}
    
    for split in ["train", "val"]:
        loss_list = []
        for _ in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data, batch_size, device)
            _, loss = model(X, Y)
            loss_list.append(loss.item())
        losses[split] = sum(loss_list) / len(loss_list)
    
    model.train()
    return losses


def format_duration(seconds: int) -> str:
    """Format seconds into human-readable duration."""
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def get_lr(step: int, max_steps: int, learning_rate: float, min_lr: float, warmup_steps: int) -> float:
    """Calculate learning rate with cosine annealing and warmup."""
    if step < warmup_steps:
        return learning_rate * max(step + 1, 1) / warmup_steps
    if step > max_steps:
        return min_lr
    
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def main():
    """Main training loop."""
    # Ensure config directories exist
    Config.ensure_directories()
    
    # Parse arguments
    args = parse_args()
    
    # Get preset configuration
    preset_config = TrainingConfig.get_preset(args.preset)
    
    # Override with command-line arguments
    max_steps = args.max_steps or preset_config["max_steps"]
    batch_size = args.batch_size or preset_config["batch_size"]
    eval_iters = args.eval_iters or preset_config["eval_iters"]
    eval_interval = args.eval_interval or preset_config["eval_interval"]
    
    # Setup logging
    LogManager.setup(
        log_file=Config.LOG_FILE,
        log_level=Config.LOG_LEVEL
    )
    
    logger.info("=" * 60)
    logger.info("MapMindGPT Training Started")
    logger.info("=" * 60)
    logger.info(f"Preset: {args.preset}")
    logger.info(f"Max steps: {max_steps}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {args.learning_rate:.2e}")
    
    try:
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device: {device}")
        if device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Get training file
        train_file = get_training_file(args.data_file)
        
        # Load tokenizer
        tokenizer = load_tokenizer(Config.VOCAB_FILE, Config.MERGES_FILE)
        
        # Load training data
        train_data, val_data = load_training_data(train_file, tokenizer, device)
        
        # Initialize model
        model = GPT(ModelConfig()).to(device)
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Load existing checkpoint or start fresh
        if args.fresh:
            logger.info("Starting training from random weights")
        elif Config.CHECKPOINT_PATH.exists():
            try:
                load_model_checkpoint(model, str(Config.CHECKPOINT_PATH), device)
                logger.info(f"Loaded checkpoint: {Config.CHECKPOINT_PATH}")
            except RuntimeError as e:
                logger.warning(f"Checkpoint incompatible, starting fresh: {e}")
        else:
            logger.info("No checkpoint found, starting from random weights")
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=0.1
        )
        scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None
        
        # Learning rate schedule
        warmup_steps = min(ModelConfig.warmup_steps, max(10, max_steps // 10))
        min_lr = ModelConfig.min_learning_rate
        
        # Training loop
        best_val = float("inf")
        start_time = time.time()
        
        logger.info("=" * 60)
        logger.info("Training loop started")
        logger.info("=" * 60)
        
        for step in range(max_steps):
            # Update learning rate
            lr = get_lr(step, max_steps, args.learning_rate, min_lr, warmup_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Get batch
            xb, yb = get_batch("train", train_data, val_data, batch_size, device)
            
            # Forward pass
            if device == "cuda":
                with torch.amp.autocast("cuda"):
                    _, loss = model(xb, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), ModelConfig.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                _, loss = model(xb, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), ModelConfig.grad_clip)
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)
            
            # Log progress
            if step % 10 == 0 or step == max_steps - 1:
                elapsed = time.time() - start_time
                done = step + 1
                progress = done / max_steps
                eta = (elapsed / progress - elapsed) if progress > 0 else 0
                
                logger.info(
                    f"Progress {done}/{max_steps} ({progress:.1%}) | "
                    f"Loss {loss.item():.4f} | "
                    f"LR {lr:.2e} | "
                    f"Elapsed {format_duration(elapsed)} | "
                    f"ETA {format_duration(eta)}"
                )
            
            # Evaluate
            if step % eval_interval == 0 or step == max_steps - 1:
                losses = estimate_loss(model, train_data, val_data, batch_size, eval_iters, device)
                logger.info(
                    f"Eval {step}: Train loss {losses['train']:.4f}, "
                    f"Val loss {losses['val']:.4f}, LR {lr:.2e}"
                )
                
                if losses["val"] < best_val:
                    best_val = losses["val"]
                    Config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
                    save_model_checkpoint(
                        model,
                        str(Config.CHECKPOINT_PATH),
                        step=step,
                        best_val=best_val,
                        train_file=str(train_file),
                        config={k: v for k, v in ModelConfig.__dict__.items() if not k.startswith("_")}
                    )
                    logger.info(
                        f"Best checkpoint saved at step {step} "
                        f"with val loss {best_val:.4f}"
                    )
        
        logger.info("=" * 60)
        logger.info("Training completed successfully")
        logger.info("=" * 60)
        logger.info(f"Best validation loss: {best_val:.4f}")
        logger.info(f"Total time: {format_duration(time.time() - start_time)}")
        logger.info(f"Checkpoint saved: {Config.CHECKPOINT_PATH}")
        
    except Exception as e:
        logger.exception("Training failed")
        raise


if __name__ == "__main__":
    main()
