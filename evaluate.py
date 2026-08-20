import argparse
import json
from pathlib import Path

from evaluation.co3d import load_co3d_scenes
from evaluation.evaluator import evaluate_scenes
from evaluation.model_runner import VGGTModelRunner
from evaluation.realestate10k import load_realestate10k_scenes


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VGGT camera pose estimation")
    parser.add_argument("--dataset", choices=["co3d", "realestate10k"], required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", default="facebook/VGGT-1B", help="Hugging Face model id or local checkpoint")
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


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

    runner = VGGTModelRunner.from_pretrained(args.model)
    result = evaluate_scenes(runner, scenes, dataset_name=dataset_name)
    result.update({"model": args.model, "num_frames": args.num_frames, "seed": args.seed})

    output = args.output or Path("results") / f"vggt_{args.dataset}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Dataset: {result['dataset']}")
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
