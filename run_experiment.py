"""CLI entry point for running model inference on Colab or locally.

Examples:
  python run_experiment.py --model blip
  python run_experiment.py --model custom --test-json data/synthetic_vlm_test.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.inference_pipeline import load_dataset, run_inference, save_experiment_log, save_results
from src.perturbation_engine import PerturbationConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run synthetic VLM robustness inference.")
    parser.add_argument("--model", choices=["blip", "custom"], required=True)
    parser.add_argument("--test-json", default="data/synthetic_vlm_test.json")
    parser.add_argument("--image-dir", default="data/images")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--brightness-factor", type=float, default=0.5)
    parser.add_argument("--noise-std", type=float, default=20.0)
    parser.add_argument("--custom-checkpoint", default="custom_dual_head_model.pth")
    parser.add_argument("--limit", type=int, default=None, help="Optional quick-test limit.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = PerturbationConfig(brightness_factor=args.brightness_factor, noise_std=args.noise_std)

    if args.model == "blip":
        from src.blip_wrapper import BLIPVQAWrapper

        wrapper = BLIPVQAWrapper()
        output_name = "blip_results_test.json"
    else:
        from src.custom_model import CustomModelConfig, CustomVQAWrapper

        wrapper = CustomVQAWrapper(CustomModelConfig(checkpoint_path=args.custom_checkpoint))
        output_name = "custom_results_test.json"

    records = load_dataset(args.test_json)
    if args.limit:
        records = records[: args.limit]

    results = run_inference(wrapper, records, args.image_dir, perturbation_config=cfg)
    output_dir = Path(args.output_dir)
    result_path = save_results(results, output_dir / output_name)
    log_path = save_experiment_log(
        output_dir / f"{args.model}_experiment_log.json",
        wrapper,
        args.test_json,
        len(records),
        cfg,
    )
    print(f"Saved results: {result_path}")
    print(f"Saved experiment log: {log_path}")


if __name__ == "__main__":
    main()
