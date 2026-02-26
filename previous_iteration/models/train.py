import torch
import torch.nn as nn
import torch.optim as optim
from loss import cross_modal_contrastive_loss


def train(
    model,
    train_loader,
    logger=None,
    *,
    epochs: int = 50,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    grad_clip: float | None = 1.0,
    device: torch.device | None = None,
    noise_fn=None,                 # callable or None, e.g., lambda x: apply_noise_injection(x, 0.05)
    feature_dropout_fn=None,       # callable or None, e.g., apply_random_feature_dropout
    dtype=torch.float64,
):
    """
    Expects each batch to be either:
      (x, y)                           -> lengths=None
      (x, y, lengths)                  -> lengths is 1D LongTensor
    All models: forward(x, lengths=None) -> logits
                forward_features(x, lengths=None) -> embedding
    """
    if dtype == torch.float64:
        model.double()
    else:
        model.float()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    if logger:
        logger.info(f"Training on device: {device}")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for batch in train_loader:
            # Unpack (x, y) or (x, y, lengths)
            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                batch_x, batch_y, lengths = batch
            else:
                batch_x, batch_y = batch
                lengths = None

            # Augment (CPU tensors are fine here)
            if noise_fn is not None:
                batch_x = noise_fn(batch_x)
            if feature_dropout_fn is not None:
                batch_x = feature_dropout_fn(batch_x)

            # Move to device
            batch_x = batch_x.to(device, dtype=torch.float32)
            batch_y = batch_y.to(device, dtype=torch.long)
            lengths = lengths.to(device) if lengths is not None else None

            optimizer.zero_grad()

            # Forward -> logits only (consistent across models)
            logits = model(batch_x, lengths=lengths)
            loss = criterion(logits, batch_y)

            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            # Stats
            bs = batch_y.size(0)
            total_loss += loss.item() * bs
            total_correct += (logits.argmax(dim=1) == batch_y).sum().item()
            total_examples += bs

        avg_loss = total_loss / max(total_examples, 1)
        acc = 100.0 * total_correct / max(total_examples, 1)
        msg = f"Epoch {epoch:02d} | loss {avg_loss:.4f} | acc {acc:.2f}%"
        (logger.info(msg) if logger else print(msg))


def contrastive_train(
    gcms_encoder,                  # must implement forward_features(x, lengths=None)
    sensor_encoder,                # must implement forward_features(x, lengths=None)
    dataloader,
    logger=None,
    *,
    temperature: float = 0.07,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    grad_clip: float | None = 1.0,
    device: torch.device | None = None,
    noise_fn=None,                 # augmentation for sensor stream
    feature_dropout_fn=None,
    dtype=torch.float64,
):
    """
    Expects each batch to be one of:
      (x_gcms, x_sensor)
      ((x_gcms, len_g), (x_sensor, len_s))
      or (x_gcms, (x_sensor, len_s))  # len_g optional, len_s optional
    """
    if dtype == torch.float64:
        gcms_encoder.double()
        sensor_encoder.double()
    else:
        gcms_encoder.float()
        sensor_encoder.float()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if logger:
        logger.info(f"Contrastive training on device: {device}")
    gcms_encoder.to(device).train()
    sensor_encoder.to(device).train()

    params = list(gcms_encoder.parameters()) + list(sensor_encoder.parameters())
    optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)

    def _unpack_modality(m):
        # Return (x, lengths or None)
        if isinstance(m, (tuple, list)) and len(m) == 2 and torch.is_tensor(m[0]):
            x, l = m
            return x, l
        return m, None

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            # Unpack GCMS and Sensor (with optional lengths)
            x_gcms, x_sensor = batch
            x_gcms, len_g = _unpack_modality(x_gcms)
            x_sensor, len_s = _unpack_modality(x_sensor)

            # Augment sensor stream if requested
            if noise_fn is not None:
                x_sensor = noise_fn(x_sensor)
            if feature_dropout_fn is not None:
                x_sensor = feature_dropout_fn(x_sensor)

            # To device
            x_gcms = x_gcms.to(device, dtype=torch.float32)
            x_sensor = x_sensor.to(device, dtype=torch.float32)
            len_g = len_g.to(device) if len_g is not None else None
            len_s = len_s.to(device) if len_s is not None else None

            optimizer.zero_grad()

            # Encodings (use the unified feature API)
            z_gcms = gcms_encoder.forward_features(x_gcms, lengths=len_g)
            z_sensor = sensor_encoder.forward_features(x_sensor, lengths=len_s)

            # Compute contrastive loss
            loss = cross_modal_contrastive_loss(z_gcms, z_sensor, temperature)

            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg = total_loss / max(num_batches, 1)
        (logger.info(f"Epoch {epoch:03d} | contrastive loss {avg:.4f}")
         if logger else print(f"Epoch {epoch:03d} | contrastive loss {avg:.4f}"))
