from pathlib import Path
import random

import torch
import torch.nn.functional as F

from .model import save_adapter_checkpoint


def alignment_loss(pred, target, mse_weight=1.0, cosine_weight=1.0):
    mse = F.mse_loss(pred, target)
    cosine = (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
    return mse_weight * mse + cosine_weight * cosine


def iter_patch_batches(cache_files, batch_size, seed=0, shuffle=True):
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    files = [Path(path) for path in cache_files]
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(files)

    generator = torch.Generator().manual_seed(seed)
    for path in files:
        record = torch.load(path, map_location="cpu")
        raw_dino3 = record["dino3"]
        raw_dino2 = record["dino2"]
        if raw_dino3.ndim != 3 or raw_dino2.ndim != 3:
            raise ValueError(f"{path}: cached DINOv2/DINOv3 tensors must be [frames, patches, dim]")
        if tuple(raw_dino3.shape[:2]) != tuple(raw_dino2.shape[:2]):
            raise ValueError(f"{path}: cached DINOv2/DINOv3 frame/patch axes do not match")
        dino3 = raw_dino3.reshape(-1, raw_dino3.shape[-1]).float()
        dino2 = raw_dino2.reshape(-1, raw_dino2.shape[-1]).float()
        indices = torch.randperm(len(dino3), generator=generator) if shuffle else torch.arange(len(dino3))
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            yield dino3[batch_indices], dino2[batch_indices]


def train_epoch(adapter, cache_files, optimizer, device, batch_size, seed, mse_weight=1.0, cosine_weight=1.0):
    device = torch.device(device)
    adapter.train()
    total_loss = 0.0
    total_items = 0
    for dino3, dino2 in iter_patch_batches(cache_files, batch_size, seed=seed, shuffle=True):
        dino3 = dino3.to(device)
        dino2 = dino2.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = adapter(dino3)
        loss = alignment_loss(pred, dino2, mse_weight=mse_weight, cosine_weight=cosine_weight)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * len(dino3)
        total_items += len(dino3)
    if total_items == 0:
        raise ValueError("No training patch pairs found")
    return total_loss / total_items


def validate_epoch(adapter, cache_files, device, batch_size, mse_weight=1.0, cosine_weight=1.0):
    device = torch.device(device)
    adapter.eval()
    total_loss = 0.0
    total_items = 0
    with torch.inference_mode():
        for dino3, dino2 in iter_patch_batches(cache_files, batch_size, seed=0, shuffle=False):
            dino3 = dino3.to(device)
            dino2 = dino2.to(device)
            pred = adapter(dino3)
            loss = alignment_loss(pred, dino2, mse_weight=mse_weight, cosine_weight=cosine_weight)
            total_loss += float(loss) * len(dino3)
            total_items += len(dino3)
    if total_items == 0:
        raise ValueError("No validation patch pairs found")
    return total_loss / total_items


def fit_adapter(adapter, train_files, val_files, output_dir, device="cpu", epochs=20, batch_size=4096,
                lr=1e-3, weight_decay=1e-4, seed=0, mse_weight=1.0, cosine_weight=1.0, metadata=None):
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    device = torch.device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter = adapter.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=weight_decay)
    base_metadata = dict(metadata or {})
    base_metadata.update({
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "seed": seed,
        "mse_weight": mse_weight,
        "cosine_weight": cosine_weight,
    })

    history = []
    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            adapter, train_files, optimizer, device, batch_size, seed=seed + epoch - 1,
            mse_weight=mse_weight, cosine_weight=cosine_weight,
        )
        val_loss = validate_epoch(
            adapter, val_files, device, batch_size,
            mse_weight=mse_weight, cosine_weight=cosine_weight,
        )
        epoch_result = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(epoch_result)
        checkpoint_metadata = dict(base_metadata)
        checkpoint_metadata.update(epoch_result)
        if val_loss < best_val:
            best_val = val_loss
            save_adapter_checkpoint(output_dir / "best_adapter.pt", adapter, checkpoint_metadata)

    final_metadata = dict(base_metadata)
    final_metadata.update(history[-1])
    save_adapter_checkpoint(output_dir / "last_adapter.pt", adapter, final_metadata)
    return history
