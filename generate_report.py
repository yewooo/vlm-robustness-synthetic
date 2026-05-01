"""Generate report-ready visualizations from existing result JSON files.

This is the fastest Colab path after uploading the folder:
  python generate_report.py

It uses the already-saved BLIP/custom prediction JSON files and creates:
- high-resolution PNG summary plots
- original-vs-perturbed failure case figures
- an HTML report with embedded images
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.metrics import find_robustness_failures
from src.perturbation_engine import PerturbationConfig
from src.visualization import (
    generate_failure_gallery,
    generate_html_report,
    load_json,
    plot_accuracy_summary,
    plot_per_task_accuracy,
    plot_robustness_degradation,
    plot_significance_table,
    save_significance_tests,
    summarize_failures,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate VLM robustness visualizations and HTML report.")
    parser.add_argument("--blip-results", default="results/blip_results_test.json")
    parser.add_argument("--custom-results", default="results/custom_results_test.json")
    parser.add_argument("--image-dir", default="data/images")
    parser.add_argument("--fig-dir", default="figs")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--brightness-factor", type=float, default=0.5)
    parser.add_argument("--noise-std", type=float, default=20.0)
    parser.add_argument("--max-failures", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    fig_dir = Path(args.fig_dir)
    report_dir = Path(args.report_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    results_by_model = {
        "BLIP VQA Base": load_json(args.blip_results),
        "Custom Dual-Head VQA": load_json(args.custom_results),
    }
    perturbation_config = PerturbationConfig(
        brightness_factor=args.brightness_factor,
        noise_std=args.noise_std,
    )

    summary_figures = [
        plot_accuracy_summary(results_by_model, fig_dir / "fig1_overall_accuracy_high_quality.png", dpi=args.dpi),
        plot_robustness_degradation(results_by_model, fig_dir / "fig2_robustness_degradation.png", dpi=args.dpi),
        plot_per_task_accuracy(results_by_model, fig_dir / "fig3_per_task_accuracy_breakdown.png", dpi=args.dpi),
        plot_significance_table(results_by_model, fig_dir / "fig4_paired_significance_tests.png", dpi=args.dpi),
    ]
    failure_summary = summarize_failures(results_by_model, report_dir / "robustness_failure_summary.json")
    significance_path = save_significance_tests(
        results_by_model,
        report_dir / "statistical_significance_tests.json",
        report_dir / "statistical_significance_tests.csv",
    )
    significance_rows = load_json(significance_path)

    failure_images = []
    for model_name, results in results_by_model.items():
        for perturbation in ["brightness", "noise", "text"]:
            if not find_robustness_failures(results, perturbation, max_examples=1):
                continue
            failure_images.extend(
                generate_failure_gallery(
                    results=results,
                    image_dir=args.image_dir,
                    output_dir=fig_dir / "failure_cases",
                    model_name=model_name,
                    perturbation=perturbation,
                    max_examples=args.max_failures,
                    perturbation_config=perturbation_config,
                    dpi=args.dpi,
                )
            )

    report_path = generate_html_report(
        results_by_model=results_by_model,
        figure_paths=summary_figures,
        failure_image_paths=failure_images,
        output_path=report_dir / "vlm_robustness_report.html",
        significance_rows=significance_rows,
    )

    print("Generated summary figures:")
    for path in summary_figures:
        print(f"  {path}")
    print(f"Generated failure summary: {failure_summary}")
    print(f"Generated significance tests: {significance_path}")
    print(f"Generated {len(failure_images)} failure-case figures in {fig_dir / 'failure_cases'}")
    print(f"Generated HTML report: {report_path}")


if __name__ == "__main__":
    main()
