"""High-quality visualizations and HTML reports for VLM robustness.

This module is designed for Colab: it reads the JSON result files you already
produce and creates report-ready PNG files plus an optional HTML report.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image, ImageOps

from .metrics import (
    PREDICTION_FIELDS,
    compute_accuracy_table,
    compute_per_task_accuracy,
    compute_paired_significance_tests,
    compute_robustness_degradation,
    find_robustness_failures,
    readable_perturbation_name,
    readable_query_type,
)
from .perturbation_engine import PerturbationConfig, apply_perturbation


PLOT_STYLE = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 11,
}


def load_json(path: str | Path):
    with open(path, "r") as f:
        return json.load(f)


def add_image_border(
    image: Image.Image,
    border_width: int = 3,
    border_color: str = "#444444",
) -> Image.Image:
    """Add a visible border so white synthetic images have clear boundaries."""

    return ImageOps.expand(image.convert("RGB"), border=border_width, fill=border_color)


def load_record_image(record: dict, image_dir: str | Path) -> Image.Image:
    return Image.open(Path(image_dir) / record["image_filename"]).convert("RGB")


def _format_correctness(is_correct: bool) -> str:
    return "Correct" if is_correct else "Incorrect"


def create_prediction_comparison_figure(
    record: dict,
    image_dir: str | Path,
    output_path: str | Path,
    model_name: str,
    perturbation: str,
    perturbation_config: Optional[PerturbationConfig] = None,
    dpi: int = 300,
    border_width: int = 3,
    border_color: str = "#444444",
) -> Path:
    """Create one side-by-side original vs perturbed prediction figure."""

    original_image = load_record_image(record, image_dir)
    perturbed_image, perturbed_query = apply_perturbation(
        original_image,
        record["query"],
        perturbation,
        config=perturbation_config,
        seed=record.get("example_index", 0),
    )

    original_image = add_image_border(original_image, border_width, border_color)
    perturbed_image = add_image_border(perturbed_image, border_width, border_color)

    clean_output = record["clean_pred"]
    perturbed_output = record[PREDICTION_FIELDS[perturbation]]
    perturbed_correct_field = {
        "brightness": "bright_correct",
        "noise": "noisy_correct",
        "text": "text_correct",
    }[perturbation]

    is_failure = record["clean_correct"] and not record[perturbed_correct_field]
    result_label = "Robustness Failure" if is_failure else "No Robustness Failure"

    fig = plt.figure(figsize=(10.5, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[4.5, 1.2, 1.7])

    ax_original = fig.add_subplot(gs[0, 0])
    ax_perturbed = fig.add_subplot(gs[0, 1])
    for ax, image, title in [
        (ax_original, original_image, "Original Image"),
        (ax_perturbed, perturbed_image, f"{readable_perturbation_name(perturbation)} Perturbation"),
    ]:
        ax.imshow(image)
        ax.set_title(title, fontsize=14, weight="bold", pad=10)
        ax.axis("off")

    ax_query = fig.add_subplot(gs[1, :])
    ax_query.axis("off")
    ax_query.text(
        0.01,
        0.75,
        f"Query: {record['query']}",
        ha="left",
        va="center",
        fontsize=12,
        wrap=True,
    )
    if perturbation == "text":
        ax_query.text(
            0.01,
            0.28,
            f"Perturbed Query: {perturbed_query}",
            ha="left",
            va="center",
            fontsize=12,
            wrap=True,
        )

    ax_result = fig.add_subplot(gs[2, :])
    ax_result.axis("off")
    details = [
        f"Model: {model_name}",
        f"Ground Truth: {record['answer']}",
        f"Clean Model Output: {clean_output} ({_format_correctness(record['clean_correct'])})",
        f"Perturbed Model Output: {perturbed_output} ({_format_correctness(record[perturbed_correct_field])})",
        f"Result: {result_label}",
    ]
    colors = ["#222222", "#222222", "#1f4e79", "#8a3ffc" if not is_failure else "#b00020", "#b00020" if is_failure else "#2f6b2f"]
    y = 0.95
    for detail, color in zip(details, colors):
        ax_result.text(0.01, y, detail, ha="left", va="top", fontsize=12, color=color)
        y -= 0.2

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def generate_failure_gallery(
    results: list[dict],
    image_dir: str | Path,
    output_dir: str | Path,
    model_name: str,
    perturbation: str,
    max_examples: int = 12,
    perturbation_config: Optional[PerturbationConfig] = None,
    dpi: int = 300,
) -> list[Path]:
    """Generate side-by-side figures for automatic robustness failures."""

    output_dir = Path(output_dir)
    failures = find_robustness_failures(results, perturbation, max_examples=max_examples)
    paths = []
    for rank, record in enumerate(failures, start=1):
        image_id = Path(record["image_filename"]).stem
        out = output_dir / f"{model_name.lower().replace(' ', '_')}_{perturbation}_failure_{rank:02d}_{image_id}.png"
        paths.append(
            create_prediction_comparison_figure(
                record=record,
                image_dir=image_dir,
                output_path=out,
                model_name=model_name,
                perturbation=perturbation,
                perturbation_config=perturbation_config,
                dpi=dpi,
            )
        )
    return paths


def plot_accuracy_summary(results_by_model: dict[str, list[dict]], output_path: str | Path, dpi: int = 300) -> Path:
    """Plot overall accuracy for each model and perturbation."""

    with plt.rc_context(PLOT_STYLE):
        df = pd.DataFrame(compute_accuracy_table(results_by_model))
        fig, ax = plt.subplots(figsize=(9.5, 5.8))
        sns.barplot(data=df, x="Perturbation", y="Accuracy", hue="Model", ax=ax, palette="Set2")
        ax.set_title("Overall VQA Accuracy Under Controlled Perturbations", fontsize=15, weight="bold")
        ax.set_xlabel("Evaluation Condition")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.0)
        ax.legend(title="Model", frameon=False, loc="upper right")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", padding=3, fontsize=9)
        fig.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return output_path


def plot_robustness_degradation(results_by_model: dict[str, list[dict]], output_path: str | Path, dpi: int = 300) -> Path:
    """Plot accuracy drop relative to clean inputs."""

    with plt.rc_context(PLOT_STYLE):
        df = pd.DataFrame(compute_robustness_degradation(results_by_model))
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        sns.lineplot(
            data=df,
            x="Perturbation",
            y="Accuracy Drop",
            hue="Model",
            marker="o",
            linewidth=2.5,
            markersize=8,
            ax=ax,
            palette="Set1",
        )
        ax.axhline(0, color="#333333", linewidth=1)
        ax.set_title("Robustness Degradation Relative to Original Inputs", fontsize=15, weight="bold")
        ax.set_xlabel("Perturbation Type")
        ax.set_ylabel("Accuracy Drop")
        ax.legend(title="Model", frameon=False)
        fig.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return output_path


def plot_per_task_accuracy(results_by_model: dict[str, list[dict]], output_path: str | Path, dpi: int = 300) -> Path:
    """Plot per-task accuracy with separate facets for each perturbation."""

    df = pd.DataFrame(compute_per_task_accuracy(results_by_model))
    with sns.axes_style("whitegrid"):
        grid = sns.catplot(
            data=df,
            kind="bar",
            x="Question Type",
            y="Accuracy",
            hue="Model",
            col="Perturbation",
            col_wrap=2,
            palette="Set2",
            height=4.0,
            aspect=1.25,
            sharey=True,
        )
        grid.set_axis_labels("Question Type", "Accuracy")
        grid.set_titles("{col_name}")
        for ax in grid.axes.flat:
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=25)
        grid.fig.suptitle("Accuracy Breakdown by Question Type", fontsize=16, weight="bold", y=1.03)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        grid.fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(grid.fig)
    return output_path


def save_significance_tests(
    results_by_model: dict[str, list[dict]],
    output_json_path: str | Path,
    output_csv_path: str | Path | None = None,
    baseline_name: str = "BLIP VQA Base",
    comparison_name: str = "Custom Dual-Head VQA",
) -> Path:
    """Save paired significance tests comparing two models."""

    rows = compute_paired_significance_tests(
        results_by_model[baseline_name],
        results_by_model[comparison_name],
        baseline_name=baseline_name,
        comparison_name=comparison_name,
    )
    output_json_path = Path(output_json_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if output_csv_path is not None:
        pd.DataFrame(rows).to_csv(output_csv_path, index=False)
    return output_json_path


def plot_significance_table(
    results_by_model: dict[str, list[dict]],
    output_path: str | Path,
    dpi: int = 300,
    baseline_name: str = "BLIP VQA Base",
    comparison_name: str = "Custom Dual-Head VQA",
) -> Path:
    """Render McNemar/CI results as a report-ready table image."""

    rows = compute_paired_significance_tests(
        results_by_model[baseline_name],
        results_by_model[comparison_name],
        baseline_name=baseline_name,
        comparison_name=comparison_name,
    )
    display_rows = []
    for row in rows:
        display_rows.append(
            {
                "Condition": row["Perturbation"],
                "BLIP Acc.": f"{row['Baseline Accuracy']:.3f}",
                "Custom Acc.": f"{row['Comparison Accuracy']:.3f}",
                "Diff.": f"{row['Accuracy Difference']:.3f}",
                "95% CI": f"[{row['Bootstrap 95% CI Low']:.3f}, {row['Bootstrap 95% CI High']:.3f}]",
                "McNemar p": f"{row['McNemar Exact P-Value']:.3f}",
            }
        )
    df = pd.DataFrame(display_rows)

    fig_height = 1.2 + 0.55 * len(df)
    fig, ax = plt.subplots(figsize=(11.5, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.35)
    for (row_idx, _), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#333333")
        else:
            cell.set_facecolor("#f7f7f7" if row_idx % 2 == 0 else "white")
    ax.set_title(
        "Paired Statistical Comparison: BLIP vs Custom Model",
        fontsize=14,
        weight="bold",
        pad=14,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def summarize_failures(results_by_model: dict[str, list[dict]], output_path: str | Path) -> Path:
    """Save a JSON summary of robustness failures for reproducibility."""

    summary = {}
    for model_name, results in results_by_model.items():
        summary[model_name] = {}
        for perturbation in ["brightness", "noise", "text"]:
            failures = find_robustness_failures(results, perturbation)
            by_type = {}
            for row in failures:
                qtype = readable_query_type(row["metadata"]["query_type"])
                by_type[qtype] = by_type.get(qtype, 0) + 1
            summary[model_name][perturbation] = {
                "num_robustness_failures": len(failures),
                "by_question_type": by_type,
            }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_path


def _image_to_base64(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode("ascii")


def generate_html_report(
    results_by_model: dict[str, list[dict]],
    figure_paths: list[str | Path],
    failure_image_paths: list[str | Path],
    output_path: str | Path,
    significance_rows: list[dict] | None = None,
) -> Path:
    """Generate a compact HTML report with figures and failure examples."""

    rows = compute_accuracy_table(results_by_model)
    accuracy_rows = "\n".join(
        f"<tr><td>{html.escape(r['Model'])}</td><td>{html.escape(r['Perturbation'])}</td><td>{r['Accuracy']:.3f}</td></tr>"
        for r in rows
    )

    figure_html = []
    for path in figure_paths:
        path = Path(path)
        figure_html.append(
            f"<section><h2>{html.escape(path.stem.replace('_', ' ').title())}</h2>"
            f"<img src='data:image/png;base64,{_image_to_base64(path)}' /></section>"
        )

    failure_html = []
    for path in failure_image_paths:
        path = Path(path)
        failure_html.append(
            f"<figure><img src='data:image/png;base64,{_image_to_base64(path)}' />"
            f"<figcaption>{html.escape(path.stem)}</figcaption></figure>"
        )

    significance_html = ""
    if significance_rows:
        sig_rows = "\n".join(
            "<tr>"
            f"<td>{html.escape(row['Perturbation'])}</td>"
            f"<td>{row['Baseline Accuracy']:.3f}</td>"
            f"<td>{row['Comparison Accuracy']:.3f}</td>"
            f"<td>{row['Accuracy Difference']:.3f}</td>"
            f"<td>[{row['Bootstrap 95% CI Low']:.3f}, {row['Bootstrap 95% CI High']:.3f}]</td>"
            f"<td>{row['McNemar Exact P-Value']:.3f}</td>"
            "</tr>"
            for row in significance_rows
        )
        significance_html = f"""
  <h2>Paired Statistical Significance Tests</h2>
  <p>Accuracy difference is BLIP accuracy minus Custom accuracy. McNemar's exact test compares paired correctness on the same examples.</p>
  <table>
    <tr><th>Condition</th><th>BLIP Accuracy</th><th>Custom Accuracy</th><th>Accuracy Difference</th><th>Bootstrap 95% CI</th><th>McNemar p-value</th></tr>
    {sig_rows}
  </table>
"""

    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Synthetic VLM Robustness Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #222; line-height: 1.45; }}
    h1, h2 {{ color: #111; }}
    table {{ border-collapse: collapse; margin: 16px 0 28px; }}
    th, td {{ border: 1px solid #bbb; padding: 8px 12px; text-align: left; }}
    th {{ background: #f2f2f2; }}
    img {{ max-width: 100%; border: 1px solid #bbb; }}
    .gallery {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 20px; }}
    figure {{ margin: 0; }}
    figcaption {{ font-size: 12px; color: #555; margin-top: 6px; }}
  </style>
</head>
<body>
  <h1>Synthetic VLM Robustness Evaluation</h1>
  <p>This report emphasizes visual evidence: original inputs, perturbed inputs, ground-truth answers, and model outputs.</p>
  <h2>Overall Accuracy</h2>
  <table>
    <tr><th>Model</th><th>Evaluation Condition</th><th>Accuracy</th></tr>
    {accuracy_rows}
  </table>
  {significance_html}
  {''.join(figure_html)}
  <h2>Automatically Identified Robustness Failures</h2>
  <div class="gallery">{''.join(failure_html)}</div>
</body>
</html>
"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")
    return output_path
