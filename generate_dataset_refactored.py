"""
Refactored generate_dataset.py with improved performance and memory efficiency.
"""
import os
import logging
from pathlib import Path
from typing import List, Tuple, Dict
from config import Config
from logging_config import LogManager, get_logger

logger = get_logger(__name__)


KNOWLEDGE_BASE: Dict[str, List[Tuple[str, str]]] = {
    "EDI": [
        ("What is EDI?", "EDI stands for Electronic Data Interchange, used for exchanging structured business documents."),
        ("What is X12?", "ANSI X12 is a common EDI standard used mainly in North America."),
        ("What is EDIFACT?", "EDIFACT is an international EDI standard."),
        ("What is AS2?", "AS2 is a protocol for securely exchanging EDI documents over HTTP."),
        ("What is an 850?", "EDI 850 is a purchase order transaction."),
        ("What is an 810?", "EDI 810 is an invoice transaction."),
        ("What is an 856?", "EDI 856 is an advance ship notice."),
        ("What is a 997?", "EDI 997 is a functional acknowledgement."),
        ("What is ISA segment?", "ISA is the interchange control header in X12."),
        ("What is GS segment?", "GS is the functional group header."),
        ("What is ST segment?", "ST identifies the transaction set."),
        ("What is partner onboarding?", "Partner onboarding means configuring a trading partner for EDI exchange."),
    ],
    "Oracle": [
        ("What is Oracle SOA Suite?", "Oracle SOA Suite is middleware for enterprise integration."),
        ("What is OIC?", "Oracle Integration Cloud is Oracle's cloud integration platform."),
        ("What is OSB?", "Oracle Service Bus is used for service virtualization and message routing."),
        ("What is BPEL?", "BPEL is used for process orchestration in Oracle SOA."),
        ("What is Mediator?", "Mediator performs routing and transformation."),
        ("What is dehydration?", "Dehydration stores process state in the database."),
        ("What is MDS?", "Metadata Services stores shared Oracle SOA artifacts."),
        ("What is Oracle AQ?", "Oracle Advanced Queuing provides message queuing."),
        ("What is JMS?", "JMS is Java Messaging Service for asynchronous messaging."),
        ("What is fault handling?", "Fault handling manages exceptions in integration flows."),
        ("What is composite?", "A composite bundles Oracle SOA components into one deployable unit."),
    ],
    "SEEBURGER": [
        ("What is SEEBURGER?", "SEEBURGER is a B2B integration and EDI platform."),
        ("What is BIS?", "Business Integration Suite is SEEBURGER's integration platform."),
        ("What is mapping designer?", "Mapping designer is used for data transformation."),
        ("What is routing?", "Routing determines message destination and processing flow."),
        ("What is message monitoring?", "Monitoring tracks message processing and errors."),
    ],
    "XML": [
        ("What is XML?", "XML is a markup language for structured hierarchical data."),
        ("What is XSD?", "XSD defines XML schema rules."),
        ("What is XPath?", "XPath selects XML nodes using expressions."),
        ("What is XSLT?", "XSLT transforms XML documents."),
        ("What is XSLT 1.0?", "XSLT 1.0 is the original XML transformation language version."),
        ("What are namespaces?", "Namespaces prevent XML element naming conflicts."),
        ("What is template matching?", "Template matching applies XSLT rules to nodes."),
        ("What is recursion in XSLT?", "Recursion is used instead of loops for repeated processing."),
    ],
    "AI": [
        ("What is AI?", "Artificial intelligence enables machines to simulate intelligent behavior."),
        ("What is machine learning?", "Machine learning learns patterns from data."),
        ("What is transformer model?", "Transformers use self-attention to process sequences."),
        ("What is tokenization?", "Tokenization converts text into smaller units."),
        ("What are embeddings?", "Embeddings represent tokens as vectors."),
        ("What is inference?", "Inference is generating predictions using a trained model."),
    ]
}

VARIATIONS = [
    "User: {q}\nAssistant: {a}\n",
    "User: Explain {q_lower}\nAssistant: {a}\n",
    "User: Can you tell me {q_lower}\nAssistant: {a}\n",
    "User: Help me understand {q_lower}\nAssistant: {a}\n",
]


def generate_dataset(
    output_file: Path,
    num_samples: int = 100000,
    batch_size: int = 1000,
    verbose: bool = True
) -> int:
    """
    Generate training dataset with streaming to disk.
    
    Args:
        output_file: Path to output corpus file
        num_samples: Total number of samples to generate
        batch_size: Batch size for writing to disk (reduces memory usage)
        verbose: Whether to log progress
        
    Returns:
        Number of samples generated
        
    Raises:
        ValueError: If output_file is invalid
        IOError: If writing to disk fails
    """
    import random
    
    output_file = Path(output_file)
    
    if not output_file.parent.exists():
        raise ValueError(f"Output directory does not exist: {output_file.parent}")
    
    logger.info(f"Generating {num_samples:,} training samples...")
    logger.info(f"Output file: {output_file}")
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            batch = []
            domains = list(KNOWLEDGE_BASE.keys())
            
            for idx in range(num_samples):
                # Select random domain and Q&A pair
                domain = random.choice(domains)
                question, answer = random.choice(KNOWLEDGE_BASE[domain])
                
                # Select random variation
                variation_template = random.choice(VARIATIONS)
                sample = variation_template.format(q=question, q_lower=question.lower(), a=answer)
                
                batch.append(sample)
                
                # Write batch to disk
                if len(batch) >= batch_size or idx == num_samples - 1:
                    f.writelines(batch)
                    batch = []
                    
                    if verbose and (idx + 1) % (batch_size * 10) == 0 or idx == num_samples - 1:
                        progress = (idx + 1) / num_samples * 100
                        logger.info(f"Progress: {idx + 1:,}/{num_samples:,} ({progress:.1f}%)")
        
        logger.info(f"✅ Dataset generated successfully: {output_file}")
        logger.info(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
        return num_samples
        
    except Exception as e:
        logger.error(f"Failed to generate dataset: {e}")
        # Clean up partial file
        if output_file.exists():
            output_file.unlink()
        raise


def validate_dataset(file_path: Path) -> Dict[str, int]:
    """
    Validate generated dataset.
    
    Args:
        file_path: Path to dataset file
        
    Returns:
        Dictionary with validation stats
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    
    stats = {
        "file_size_mb": file_path.stat().st_size / 1024 / 1024,
        "lines": 0,
        "user_lines": 0,
        "assistant_lines": 0,
        "empty_lines": 0,
    }
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            stats["lines"] += 1
            if line.startswith("User:"):
                stats["user_lines"] += 1
            elif line.startswith("Assistant:"):
                stats["assistant_lines"] += 1
            elif not line.strip():
                stats["empty_lines"] += 1
    
    logger.info("Dataset validation:")
    logger.info(f"  Total lines: {stats['lines']:,}")
    logger.info(f"  User lines: {stats['user_lines']:,}")
    logger.info(f"  Assistant lines: {stats['assistant_lines']:,}")
    logger.info(f"  Empty lines: {stats['empty_lines']:,}")
    logger.info(f"  File size: {stats['file_size_mb']:.2f} MB")
    
    return stats


def main():
    """Main entry point."""
    Config.ensure_directories()
    LogManager.setup(
        log_file=Config.LOG_FILE,
        log_level=Config.LOG_LEVEL
    )
    
    try:
        # Generate dataset
        generated = generate_dataset(
            output_file=Config.CORPUS_FILE,
            num_samples=100000,
            batch_size=1000,
            verbose=True
        )
        
        # Validate
        stats = validate_dataset(Config.CORPUS_FILE)
        
        logger.info("=" * 60)
        logger.info("Dataset generation complete")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
