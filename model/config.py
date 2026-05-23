class Config:
    vocab_size = 32000
    block_size = 64
    n_embd = 128
    n_head = 4
    n_layer = 2
    dropout = 0.1
    pad_token_id = 0
    unk_token_id = 1
    bos_token_id = 2
    eos_token_id = 3
    learning_rate = 6e-4
    min_learning_rate = 6e-5
    batch_size = 32
    max_steps = 5000
    warmup_steps = 200
    grad_clip = 1.0
