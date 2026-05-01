"""Perturbation utilities for synthetic VLM robustness experiments.

The functions in this file are intentionally small and explicit so the final
report can describe exactly what each perturbation does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance


@dataclass(frozen=True)
class PerturbationConfig:
    """Configuration for controlled image and text perturbations."""

    brightness_factor: float = 0.5
    noise_std: float = 20.0


def perturb_brightness(image: Image.Image, factor: float = 0.5) -> Image.Image:
    """Darken or brighten an image by multiplying pixel brightness."""

    return ImageEnhance.Brightness(image).enhance(factor)


def perturb_noise(
    image: Image.Image,
    noise_std: float = 20.0,
    seed: Optional[int] = None,
) -> Image.Image:
    """Add zero-mean Gaussian noise to RGB pixels."""

    rng = np.random.default_rng(seed)
    arr = np.asarray(image).astype(np.float32)
    noise = rng.normal(0, noise_std, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def perturb_text_rephrase(query: str) -> str:
    """Apply deterministic wording changes while preserving the answer."""

    replacements = {
        "Is there a": "Does the image contain a",
        "How many": "What number of",
        "are present?": "are in the image?",
        "Are there more": "Does the image contain more",
    }
    new_query = query
    for old, new in replacements.items():
        new_query = new_query.replace(old, new)
    return new_query


def apply_perturbation(
    image: Image.Image,
    query: str,
    perturbation: str,
    config: PerturbationConfig | None = None,
    seed: Optional[int] = None,
) -> tuple[Image.Image, str]:
    """Return a perturbed image/query pair for a named perturbation."""

    cfg = config or PerturbationConfig()
    perturbation = (perturbation or "clean").lower()

    if perturbation == "clean":
        return image.copy(), query
    if perturbation == "brightness":
        return perturb_brightness(image, cfg.brightness_factor), query
    if perturbation == "noise":
        return perturb_noise(image, cfg.noise_std, seed=seed), query
    if perturbation == "text":
        return image.copy(), perturb_text_rephrase(query)

    raise ValueError(f"Unknown perturbation: {perturbation}")


def perturbation_description(perturbation: str, config: PerturbationConfig | None = None) -> str:
    """Human-readable perturbation description for reports and logs."""

    cfg = config or PerturbationConfig()
    if perturbation == "brightness":
        return f"Brightness perturbation, factor={cfg.brightness_factor}"
    if perturbation == "noise":
        return f"Gaussian noise perturbation, std={cfg.noise_std}"
    if perturbation == "text":
        return "Text rephrasing perturbation"
    if perturbation == "clean":
        return "Original image and original query"
    return perturbation
