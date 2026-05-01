"""Reusable inference pipeline for clean and perturbed VQA evaluation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from .metrics import is_correct_prediction
from .perturbation_engine import PerturbationConfig, apply_perturbation, perturbation_description


def load_dataset(json_path: str | Path) -> list[dict]:
    with open(json_path, "r") as f:
        return json.load(f)


def load_image(image_dir: str | Path, filename: str) -> Image.Image:
    return Image.open(Path(image_dir) / filename).convert("RGB")


def run_inference(
    model_wrapper,
    records: list[dict],
    image_dir: str | Path,
    perturbation_config: PerturbationConfig | None = None,
) -> list[dict]:
    """Run clean, brightness, noise, and text inference for every record."""

    cfg = perturbation_config or PerturbationConfig()
    results = []
    for idx, sample in enumerate(records):
        image = load_image(image_dir, sample["image_filename"])
        query = sample["query"]
        answer = str(sample["answer"])
        query_type = sample["metadata"]["query_type"]

        clean_pred = _predict(model_wrapper, image, query, query_type)
        bright_image, bright_query = apply_perturbation(image, query, "brightness", cfg, seed=idx)
        noisy_image, noisy_query = apply_perturbation(image, query, "noise", cfg, seed=idx)
        text_image, text_query = apply_perturbation(image, query, "text", cfg, seed=idx)

        bright_pred = _predict(model_wrapper, bright_image, bright_query, query_type)
        noisy_pred = _predict(model_wrapper, noisy_image, noisy_query, query_type)
        text_pred = _predict(model_wrapper, text_image, text_query, query_type)

        results.append(
            {
                "image_filename": sample["image_filename"],
                "query": query,
                "rephrased_query": text_query,
                "answer": answer,
                "metadata": sample["metadata"],
                "clean_pred": clean_pred,
                "bright_pred": bright_pred,
                "noisy_pred": noisy_pred,
                "text_pred": text_pred,
                "clean_correct": is_correct_prediction(clean_pred, answer),
                "bright_correct": is_correct_prediction(bright_pred, answer),
                "noisy_correct": is_correct_prediction(noisy_pred, answer),
                "text_correct": is_correct_prediction(text_pred, answer),
            }
        )
    return results


def _predict(model_wrapper, image: Image.Image, query: str, query_type: str) -> str:
    try:
        return model_wrapper.predict(image, query, query_type=query_type)
    except TypeError:
        return model_wrapper.predict(image, query)


def save_results(results: list[dict], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return output_path


def save_experiment_log(
    output_path: str | Path,
    model_wrapper,
    dataset_path: str | Path,
    num_examples: int,
    perturbation_config: PerturbationConfig,
) -> Path:
    log = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_wrapper.experiment_config(),
        "dataset_path": str(dataset_path),
        "num_examples": num_examples,
        "perturbations": {
            name: perturbation_description(name, perturbation_config)
            for name in ["clean", "brightness", "noise", "text"]
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    return output_path
