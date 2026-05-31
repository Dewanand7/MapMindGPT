"""
Configuration module for MapMindGPT.
Centralized settings management with environment variable support.
"""
import os
from pathlib import Path
from typing import Optional


class Config:
    """Main configuration class for MapMindGPT."""
    
    # ============= DIRECTORIES =============
    BASE_DIR = Path(os.getenv("BASE_DIR", "."))
    DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
    CHECKPOINTS_DIR = Path(os.getenv("CHECKPOINTS_DIR", BASE_DIR / "checkpoints"))
    TOKENIZER_DIR = Path(os.getenv("TOKENIZER_DIR", BASE_DIR / "tokenizer"))
    DOCS_ROOT = Path(os.getenv("DOCS_ROOT", BASE_DIR / "docs"))
    UPLOADS_DIR = DOCS_ROOT / "uploads"
    
    # ============= FILE PATHS =============
    CHECKPOINT_PATH = CHECKPOINTS_DIR / "model.pt"
    INDEX_FILE = DATA_DIR / "vector.index"
    META_FILE = DATA_DIR / "chunks.pkl"
    CACHE_FILE = DATA_DIR / "response_cache.json"
    LEGACY_CACHE_FILE = DATA_DIR / "response_cache.pkl"
    FEEDBACK_FILE = DATA_DIR / "feedback.jsonl"
    AUDIT_LOG_FILE = DATA_DIR / "audit_log.jsonl"
    MANUAL_QA_FILE = DATA_DIR / "manual_qa.json"
    VOCAB_FILE = TOKENIZER_DIR / "vocab.json"
    MERGES_FILE = TOKENIZER_DIR / "merges.txt"
    CORPUS_FILE = DATA_DIR / "corpus.txt"
    INSTRUCTION_CORPUS_FILE = DATA_DIR / "instruction_corpus.txt"
    FEEDBACK_TRAIN_FILE = DATA_DIR / "feedback_instruction_corpus.txt"
    EVAL_QUESTIONS_FILE = DATA_DIR / "eval_questions.json"
    
    # ============= MODEL SETTINGS =============
    # GPT Model Architecture
    VOCAB_SIZE = 50257
    BLOCK_SIZE = 1024
    N_EMBD = 384
    N_HEAD = 6
    N_LAYER = 6
    DROPOUT = 0.1
    
    # Training
    MAX_STEPS = 1200
    BATCH_SIZE = 16
    LEARNING_RATE = 6e-4
    MIN_LEARNING_RATE = 6e-5
    WARMUP_STEPS = 100
    GRAD_CLIP = 1.0
    EVAL_INTERVAL = 100
    
    # ============= DEVICE SETTINGS =============
    DEVICE = os.getenv("DEVICE", "auto")  # "auto", "cuda", "cpu"
    
    # ============= EMBEDDING SETTINGS =============
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # ============= RAG SETTINGS =============
    TOP_K_SEMANTIC = 200  # Initial retrieval
    TOP_K_FINAL = 3  # Final results
    ML_DOMAIN_THRESHOLD = 0.34
    ML_INTENT_THRESHOLD = 0.33
    MAX_CONTEXT_TOKENS = 2000
    CUSTOM_MODEL_MAX_TOKENS = 1200
    OLLAMA_MAX_TOKENS = 2000
    
    # ============= FILE UPLOAD SETTINGS =============
    MAX_UPLOAD_SIZE_MB = 25
    MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    SUPPORTED_FILE_TYPES = {
        # Text formats
        ".txt", ".md", ".markdown", ".log", ".edi", ".x12", ".dat", ".yaml", ".yml",
        # Structured formats
        ".csv", ".json", ".html", ".htm", ".xml",
        # Document formats
        ".pdf", ".docx", ".xlsx"
    }
    
    # ============= CACHE SETTINGS =============
    CACHE_MAX_ITEMS = 500
    CACHE_VERSION = "v6"
    
    # ============= LOGGING SETTINGS =============
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = DATA_DIR / "app.log"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # ============= STREAMLIT MODELS =============
    AVAILABLE_MODELS = [
        'MapMindGPT-Custom',
        'qwen2.5:7b-instruct',
        'llama3:8b',
        'mistral:7b',
        'deepseek-coder:6.7b',
        'codellama:7b'
    ]
    
    AVAILABLE_MODES = [
        'General Chat',
        'Code Assistant',
        'EDI Expert',
        'Oracle Expert',
        'XSLT/XPath Expert',
        'SEEBURGER Expert'
    ]
    
    # ============= DOMAINS & INTENTS =============
    DOMAINS = {
        'edi': ['edi', '850', '810', '856', '997', 'as2', 'x12', 'edifact'],
        'oracle': ['soa', 'oic', 'osb', 'bpel', 'mediator', 'oracle'],
        'xslt': ['xslt', 'xpath', 'xml', 'xsd', 'namespace'],
        'seeburger': ['seeburger', 'bis'],
        'ai': ['ai', 'transformer', 'llm', 'machine', 'learning']
    }
    
    INTENTS = {
        'code_example': ['code', 'example', 'syntax', 'sample'],
        'definition': ['what is', 'define', 'meaning', 'explain'],
        'troubleshooting': ['error', 'exception', 'failed', 'fix'],
        'document_lookup': ['document', 'source', 'upload', 'file'],
        'training': ['train', 'checkpoint', 'loss', 'eval'],
        'general_chat': ['hello', 'hi', 'your name', 'who are you']
    }
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Create all required directories if they don't exist."""
        directories = [
            cls.DATA_DIR,
            cls.CHECKPOINTS_DIR,
            cls.TOKENIZER_DIR,
            cls.DOCS_ROOT,
            cls.UPLOADS_DIR,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def to_dict(cls) -> dict:
        """Convert config to dictionary, excluding private attributes."""
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }
    
    @classmethod
    def validate(cls) -> list:
        """
        Validate configuration. Returns list of warnings/errors.
        """
        issues = []
        
        # Check if required directories can be created
        try:
            cls.ensure_directories()
        except Exception as e:
            issues.append(f"Cannot create directories: {e}")
        
        # Check model file existence
        if not cls.CHECKPOINT_PATH.exists():
            issues.append(f"Checkpoint not found: {cls.CHECKPOINT_PATH}")
        
        if not cls.VOCAB_FILE.exists():
            issues.append(f"Tokenizer vocab not found: {cls.VOCAB_FILE}")
        
        # Check device availability
        if cls.DEVICE == "auto":
            try:
                import torch
                cls.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                issues.append("PyTorch not installed")
        
        return issues


class TrainingConfig:
    """Training-specific configuration."""
    
    PRESETS = {
        "quick": {
            "max_steps": 300,
            "batch_size": 8,
            "eval_iters": 5,
            "eval_interval": 50,
        },
        "balanced": {
            "max_steps": 1200,
            "batch_size": 16,
            "eval_iters": 10,
            "eval_interval": 100,
        },
        "full": {
            "max_steps": Config.MAX_STEPS,
            "batch_size": Config.BATCH_SIZE,
            "eval_iters": 20,
            "eval_interval": 100,
        },
    }
    
    @classmethod
    def get_preset(cls, preset_name: str) -> dict:
        """Get preset configuration."""
        if preset_name not in cls.PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(cls.PRESETS.keys())}")
        return cls.PRESETS[preset_name].copy()


class StopWords:
    """Common stopwords for keyword extraction."""
    
    COMMON = {
        'what', 'is', 'the', 'a', 'an', 'how', 'explain', 'tell', 'me', 'about',
        'can', 'you', 'please', 'to', 'of', 'in', 'on', 'at', 'for', 'with',
        'from', 'by', 'and', 'or', 'be', 'are', 'have', 'has', 'do', 'does',
        'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must'
    }


if __name__ == "__main__":
    # Test configuration
    Config.ensure_directories()
    issues = Config.validate()
    
    if issues:
        print("⚠️ Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ Configuration valid")
        print(f"\nBase Directory: {Config.BASE_DIR}")
        print(f"Data Directory: {Config.DATA_DIR}")
        print(f"Device: {Config.DEVICE}")
