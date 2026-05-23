import torch


def load_model_checkpoint(model, path, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    else:
        state_dict = checkpoint

    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        incompatible = model.load_state_dict(state_dict, strict=False)
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)

        allowed_missing = {key for key in missing if key.endswith(".tril")}
        if missing != allowed_missing or unexpected:
            raise

    return checkpoint


def save_model_checkpoint(model, path, **metadata):
    payload = {
        "model_state_dict": model.state_dict(),
        **metadata,
    }
    torch.save(payload, path)
