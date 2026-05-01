# Probing VLM Robustness with Controlled Synthetic Data

This project evaluates vision-language model robustness on controlled synthetic VQA data. It compares a pretrained BLIP baseline with a custom task-specific dual-head VQA model under original and perturbed image/text inputs.

## Project Structure

```text
.
├── 01_data_generation.ipynb
├── 02_model_and_perturbation.ipynb
├── 03_evaluation_visualization_report.ipynb
├── config.yaml
├── src/
├── results/
├── figs/
├── reports/
└── VLM_Robustness_ProjectDeck_7slides.pptx
```

## Models

- **BLIP VQA Base:** `Salesforce/blip-vqa-base`, inference only.
- **Custom Dual-Head VQA:** ResNet-18 image encoder, bidirectional GRU text encoder, query-type embedding, shared fusion MLP, yes/no head, and counting head.

## Dataset and Evaluation

- 3000-example training pool
- 20% validation split for custom-model checkpoint selection
- 600 held-out test examples
- Query types: existence, counting, spatial relationship, comparison
- Perturbations: brightness, noise, text rephrasing

## Final Results

| Model | Original | Brightness | Noise | Text Rephrasing |
|---|---:|---:|---:|---:|
| BLIP VQA Base | 0.593 | 0.590 | 0.570 | 0.570 |
| Custom Dual-Head VQA | 0.572 | 0.573 | 0.577 | 0.568 |

Paired McNemar tests show no statistically significant overall difference between BLIP and the custom model at the 0.05 level across evaluation conditions.

## Running in Colab

1. Upload the project folder to Google Drive.
2. Open and run:

```text
01_data_generation.ipynb
02_model_and_perturbation.ipynb
03_evaluation_visualization_report.ipynb
```

The third notebook regenerates high-resolution figures and report artifacts from the saved result JSON files.

## Important Outputs

- `figs/fig1_overall_accuracy_high_quality.png`
- `figs/fig2_robustness_degradation.png`
- `figs/fig3_per_task_accuracy_breakdown.png`
- `figs/fig4_paired_significance_tests.png`
- `reports/statistical_significance_tests.json`
- `reports/statistical_significance_tests.csv`

Large generated artifacts are intentionally excluded from GitHub by `.gitignore`, including model checkpoints, generated image folders, full HTML reports, and failure-case galleries.

