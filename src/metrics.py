"""Metrics for VLM robustness evaluation."""

from __future__ import annotations

from collections import defaultdict
import math
import random
from typing import Iterable


RESULT_FIELDS = {
    "clean": "clean_correct",
    "brightness": "bright_correct",
    "noise": "noisy_correct",
    "text": "text_correct",
}

PREDICTION_FIELDS = {
    "clean": "clean_pred",
    "brightness": "bright_pred",
    "noise": "noisy_pred",
    "text": "text_pred",
}


def normalize_prediction(value) -> str:
    """Normalize an answer or prediction for exact-match scoring."""

    if value is None:
        return ""
    return str(value).strip().lower().replace(".", "").replace("?", "").replace("!", "")


def is_correct_prediction(prediction, answer) -> bool:
    """Exact-match correctness after light normalization."""

    return normalize_prediction(prediction) == normalize_prediction(answer)


def mean_bool(values: Iterable[bool]) -> float:
    values = list(values)
    return sum(bool(v) for v in values) / len(values) if values else 0.0


def compute_accuracy(results: list[dict], perturbation: str = "clean") -> float:
    """Compute accuracy for one perturbation condition."""

    field = RESULT_FIELDS[perturbation]
    return mean_bool(row[field] for row in results)


def compute_accuracy_table(results_by_model: dict[str, list[dict]]) -> list[dict]:
    """Return tidy rows with model, perturbation, and accuracy."""

    rows = []
    for model_name, results in results_by_model.items():
        for perturbation in RESULT_FIELDS:
            rows.append(
                {
                    "Model": model_name,
                    "Perturbation": readable_perturbation_name(perturbation),
                    "Accuracy": compute_accuracy(results, perturbation),
                }
            )
    return rows


def compute_per_task_accuracy(results_by_model: dict[str, list[dict]]) -> list[dict]:
    """Return tidy per-query-type accuracy rows."""

    rows = []
    for model_name, results in results_by_model.items():
        grouped = defaultdict(list)
        for row in results:
            grouped[row["metadata"]["query_type"]].append(row)
        for query_type, subset in sorted(grouped.items()):
            for perturbation in RESULT_FIELDS:
                rows.append(
                    {
                        "Model": model_name,
                        "Question Type": readable_query_type(query_type),
                        "Perturbation": readable_perturbation_name(perturbation),
                        "Accuracy": compute_accuracy(subset, perturbation),
                    }
                )
    return rows


def compute_robustness_degradation(results_by_model: dict[str, list[dict]]) -> list[dict]:
    """Compute clean accuracy minus perturbed accuracy."""

    rows = []
    for model_name, results in results_by_model.items():
        clean_acc = compute_accuracy(results, "clean")
        for perturbation in ["brightness", "noise", "text"]:
            perturbed_acc = compute_accuracy(results, perturbation)
            rows.append(
                {
                    "Model": model_name,
                    "Perturbation": readable_perturbation_name(perturbation),
                    "Clean Accuracy": clean_acc,
                    "Perturbed Accuracy": perturbed_acc,
                    "Accuracy Drop": clean_acc - perturbed_acc,
                }
            )
    return rows


def _two_sided_exact_binomial_p_value(k: int, n: int, p: float = 0.5) -> float:
    """Two-sided exact binomial p-value for small paired tests.

    This is used for McNemar's exact test. Under the null hypothesis, discordant
    pairs are equally likely to favor either model.
    """

    if n <= 0:
        return 1.0
    observed = min(k, n - k)
    tail = 0.0
    for i in range(observed + 1):
        tail += math.comb(n, i) * (p**i) * ((1 - p) ** (n - i))
    return min(1.0, 2.0 * tail)


def _paired_rows(
    baseline_results: list[dict],
    comparison_results: list[dict],
) -> list[tuple[dict, dict]]:
    """Align two result files by image filename and query."""

    comparison_index = {
        (row["image_filename"], row["query"]): row for row in comparison_results
    }
    pairs = []
    for baseline_row in baseline_results:
        key = (baseline_row["image_filename"], baseline_row["query"])
        if key in comparison_index:
            pairs.append((baseline_row, comparison_index[key]))
    return pairs


def bootstrap_accuracy_difference_ci(
    baseline_correct: list[bool],
    comparison_correct: list[bool],
    num_samples: int = 5000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap CI for paired accuracy difference.

    Difference is baseline accuracy minus comparison accuracy.
    """

    if len(baseline_correct) != len(comparison_correct):
        raise ValueError("Paired correctness lists must have the same length.")
    n = len(baseline_correct)
    if n == 0:
        return 0.0, 0.0

    rng = random.Random(seed)
    diffs = []
    for _ in range(num_samples):
        total = 0.0
        for _ in range(n):
            idx = rng.randrange(n)
            total += int(baseline_correct[idx]) - int(comparison_correct[idx])
        diffs.append(total / n)
    diffs.sort()
    lower_idx = int((alpha / 2) * num_samples)
    upper_idx = int((1 - alpha / 2) * num_samples) - 1
    return diffs[lower_idx], diffs[upper_idx]


def compute_paired_significance_tests(
    baseline_results: list[dict],
    comparison_results: list[dict],
    baseline_name: str = "BLIP VQA Base",
    comparison_name: str = "Custom Dual-Head VQA",
    num_bootstrap_samples: int = 5000,
    seed: int = 42,
) -> list[dict]:
    """Compare two models with paired McNemar tests and bootstrap CIs."""

    pairs = _paired_rows(baseline_results, comparison_results)
    rows = []
    for perturbation, field in RESULT_FIELDS.items():
        baseline_correct = [bool(b[field]) for b, _ in pairs]
        comparison_correct = [bool(c[field]) for _, c in pairs]
        n = len(pairs)
        baseline_only = sum(b and not c for b, c in zip(baseline_correct, comparison_correct))
        comparison_only = sum(c and not b for b, c in zip(baseline_correct, comparison_correct))
        both_correct = sum(b and c for b, c in zip(baseline_correct, comparison_correct))
        both_wrong = sum((not b) and (not c) for b, c in zip(baseline_correct, comparison_correct))
        discordant = baseline_only + comparison_only
        p_value = _two_sided_exact_binomial_p_value(baseline_only, discordant)
        baseline_acc = sum(baseline_correct) / n if n else 0.0
        comparison_acc = sum(comparison_correct) / n if n else 0.0
        ci_low, ci_high = bootstrap_accuracy_difference_ci(
            baseline_correct,
            comparison_correct,
            num_samples=num_bootstrap_samples,
            seed=seed,
        )
        rows.append(
            {
                "Perturbation": readable_perturbation_name(perturbation),
                "Baseline Model": baseline_name,
                "Comparison Model": comparison_name,
                "Num Paired Examples": n,
                "Baseline Accuracy": baseline_acc,
                "Comparison Accuracy": comparison_acc,
                "Accuracy Difference": baseline_acc - comparison_acc,
                "Bootstrap 95% CI Low": ci_low,
                "Bootstrap 95% CI High": ci_high,
                "Both Correct": both_correct,
                "Both Incorrect": both_wrong,
                "Baseline Correct Only": baseline_only,
                "Comparison Correct Only": comparison_only,
                "Discordant Pairs": discordant,
                "McNemar Exact P-Value": p_value,
                "Significant at 0.05": p_value < 0.05,
            }
        )
    return rows


def find_robustness_failures(
    results: list[dict],
    perturbation: str,
    max_examples: int | None = None,
) -> list[dict]:
    """Find examples that are correct when clean and incorrect after perturbation."""

    field = RESULT_FIELDS[perturbation]
    pred_field = PREDICTION_FIELDS[perturbation]
    failures = []

    for idx, row in enumerate(results):
        if row["clean_correct"] and not row[field]:
            item = dict(row)
            item["example_index"] = idx
            item["perturbation"] = perturbation
            item["perturbed_prediction"] = row[pred_field]
            item["failure_label"] = "Robustness Failure"
            failures.append(item)

    return failures[:max_examples] if max_examples else failures


def readable_query_type(query_type: str) -> str:
    mapping = {
        "existence": "Existence",
        "count": "Counting",
        "spatial": "Spatial Relationship",
        "comparison": "Comparison",
    }
    return mapping.get(query_type, query_type.title())


def readable_perturbation_name(perturbation: str) -> str:
    mapping = {
        "clean": "Original",
        "brightness": "Brightness",
        "noise": "Noise",
        "text": "Text Rephrasing",
    }
    return mapping.get(perturbation, perturbation.title())
