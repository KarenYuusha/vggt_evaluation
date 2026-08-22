import argparse
import json
from pathlib import Path

import torch

from adapter.model import build_linear_adapter, build_residual_mlp_adapter
from adapter.training import fit_adapter


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train DINOv3-to-DINOv2 feature alignment adapter")
    parser.add_argument("--cache-dir", type=Path, default=Path("adapter_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("adapter_checkpoints"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--cosine-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adapter-type", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def adapter_dims_from_manifest(manifest):
    legacy_dim = int(manifest.get("feature_dim") or 1024)
    input_dim = int(manifest.get("adapter_input_dim", manifest.get("dino3_feature_dim", legacy_dim)))
    output_dim = int(manifest.get("adapter_output_dim", manifest.get("dino2_feature_dim", legacy_dim)))
    return input_dim, output_dim


def build_adapter_from_manifest(manifest, adapter_type, hidden_dim=2048):
    input_dim, output_dim = adapter_dims_from_manifest(manifest)
    if adapter_type == "linear":
        return build_linear_adapter(
            input_dim=input_dim,
            output_dim=output_dim,
            identity_if_possible=input_dim == output_dim,
        )
    if adapter_type == "mlp":
        return build_residual_mlp_adapter(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
        )
    raise ValueError(f"Unsupported adapter type: {adapter_type}")


def main(argv=None):
    args = parse_args(argv)
    manifest_path = args.cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Cache manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_files = sorted((args.cache_dir / "train").glob("*.pt"))
    val_files = sorted((args.cache_dir / "val").glob("*.pt"))
    if len(train_files) != manifest["num_train_clips"] or len(val_files) != manifest["num_val_clips"]:
        raise ValueError(
            f"Cache file counts do not match manifest: train {len(train_files)}/{manifest['num_train_clips']}, "
            f"val {len(val_files)}/{manifest['num_val_clips']}"
        )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    adapter = build_adapter_from_manifest(manifest, args.adapter_type, hidden_dim=args.hidden_dim)
    input_dim, output_dim = adapter_dims_from_manifest(manifest)
    metadata = {
        "cache_manifest": str(manifest_path),
        "vggt_model": manifest.get("vggt_model"),
        "dinov3_model": manifest.get("dinov3_model"),
        "split_seed": manifest.get("seed"),
        "feature_mode": manifest.get("feature_mode", "final"),
        "dino3_layer_indices": manifest.get("dino3_layer_indices"),
        "adapter_input_dim": input_dim,
        "adapter_output_dim": output_dim,
        "adapter_type": args.adapter_type,
        "hidden_dim": args.hidden_dim if args.adapter_type == "mlp" else None,
    }
    history = fit_adapter(
        adapter, train_files, val_files, args.output_dir, device=device, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay, seed=args.seed,
        mse_weight=args.mse_weight, cosine_weight=args.cosine_weight, metadata=metadata,
    )
    for item in history:
        print(f"Epoch {item['epoch']:02d}: train={item['train_loss']:.6f} val={item['val_loss']:.6f}")
    print(f"Best checkpoint: {args.output_dir / 'best_adapter.pt'}")


if __name__ == "__main__":
    main()
