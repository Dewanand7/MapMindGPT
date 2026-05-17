from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    files=["data/corpus.txt"],
    vocab_size=32000,
    min_frequency=2,
    special_tokens=[
        "<pad>",
        "<unk>",
        "<bos>",
        "<eos>"
    ]
)

tokenizer.save_model("tokenizer")