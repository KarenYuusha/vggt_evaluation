import argparse
import json
from pathlib import Path

from evaluation.co3d import load_co3d_scenes
from evaluation.evaluator import evaluate_scenes
from evaluation.model_runner import VGGTModelRunner
from evaluation.realestate10k import load_realestate10k_scenes


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate VGGT camera pose estimation")
    parser.add_argument("--dataset", choices=["co3d", "realestate10k"], required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", default="facebook/VGGT-1B", help="Hugging Face model id or local checkpoint")
    parser.add_argument(
        "--backbone",
        choices=["dinov2", "dinov3", "dinov3-final", "dinov3-multilayer"],
        default="dinov2",
    )
    parser.add_argument("--adapter-checkpoint", type=Path, default=None)
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.adapter_checkpoint is not None and args.backbone == "dinov2":
        parser.error("--adapter-checkpoint requires a DINOv3 backbone")
    if args.backbone == "dinov3-multilayer" and args.adapter_checkpoint is None:
        parser.error("--backbone dinov3-multilayer requires --adapter-checkpoint")
    return args


def main():
    args = parse_args()

    if args.dataset == "co3d":
        scenes = load_co3d_scenes(args.data_dir, num_frames=args.num_frames, seed=args.seed)
        dataset_name = "CO3Dv2-single-sequence"
    else:
        scenes = load_realestate10k_scenes(args.data_dir, num_frames=args.num_frames)
        dataset_name = "RealEstate10k"

    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]
    if not scenes:
        raise RuntimeError(f"No valid scenes found in {args.data_dir}")

    runner = VGGTModelRunner.from_pretrained(
        args.model, backbone=args.backbone, adapter_checkpoint=args.adapter_checkpoint
    )
    result = evaluate_scenes(runner, scenes, dataset_name=dataset_name)
    result.update({
        "model": args.model,
        "backbone": runner.backbone,
        "adapter_checkpoint": str(args.adapter_checkpoint) if args.adapter_checkpoint is not None else None,
        "input_target_size": runner.input_target_size,
        "patch_size": runner.patch_size,
        "num_frames": args.num_frames,
        "seed": args.seed,
    })

    output = args.output or Path("results") / f"vggt_{args.dataset}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Dataset: {result['dataset']}")
    print(f"Backbone: {result['backbone']} ({result['input_target_size']}px, patch {result['patch_size']})")
    if result["adapter_checkpoint"]:
        print(f"Adapter: {result['adapter_checkpoint']}")
    print(f"Scenes: {result['num_scenes']}")
    print(f"AUC@30: {result['auc30']:.4f} ({result['auc30'] * 100:.2f}%)")
    print(f"Average inference: {result['avg_inference_ms']:.2f} ms")
    print(f"Inference std: {result['std_inference_ms']:.2f} ms")
    print(f"Average preprocessing: {result['avg_preprocess_ms']:.2f} ms")
    print(f"Average total scene: {result['avg_total_scene_ms']:.2f} ms")
    print(f"Total evaluation: {result['total_evaluation_s']:.2f} s")
    if "category_auc30" in result:
        print("Category AUC@30:")
        for category, auc30 in result["category_auc30"].items():
            print(f"  {category}: {auc30:.4f} ({auc30 * 100:.2f}%)")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
