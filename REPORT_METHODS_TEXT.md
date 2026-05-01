# Methodology Text for Final Report

## Dataset

We evaluate VLM robustness using a controlled synthetic visual-question-answering dataset. Each image contains simple geometric objects drawn on a white background. Objects vary by shape (`square`, `circle`, `triangle`), color (`red`, `blue`, `green`, `yellow`), position, and size. Each sample contains an image, a natural-language query, a ground-truth answer, and metadata describing the query type.

The dataset contains 3000 training-pool examples and 600 final test examples. For the custom model, 20% of the training pool is held out as a validation set for checkpoint selection. The final test set is reserved for reporting clean and perturbed performance.

The evaluation covers four question types:

- **Existence:** whether a target object appears in the scene.
- **Counting:** how many objects match a target color and shape.
- **Spatial relationship:** whether one object is left of or above another object.
- **Comparison:** whether one object category appears more often than another.

## Perturbations

We evaluate each model on the original input and three controlled perturbations:

- **Brightness perturbation:** the input image brightness is multiplied by `0.5`.
- **Noise perturbation:** zero-mean Gaussian noise with standard deviation `20` is added to RGB pixels.
- **Text rephrasing:** the question wording is deterministically changed while preserving the intended answer.

These perturbations allow us to measure whether a model changes its prediction when the visual or textual input is modified without changing the semantic answer.

## BLIP Baseline

The baseline model is **BLIP VQA Base**, using the Hugging Face checkpoint:

`Salesforce/blip-vqa-base`

We use the pretrained model without fine-tuning. For each image-query pair, the image and query are processed with the corresponding BLIP processor, and the model generates a short text answer. The generated answer is normalized and compared against the ground-truth answer using exact match.

## Custom Dual-Head Model

The custom model is a task-specific VQA classifier designed for the synthetic dataset. Unlike BLIP, it does not generate open-ended text. Instead, it predicts over the limited answer space defined by the dataset.

The architecture contains:

- **Image encoder:** ResNet-18 pretrained on ImageNet, with the final classification layer removed.
- **Text encoder:** learned word embeddings followed by a bidirectional GRU.
- **Question-type embedding:** a learned embedding for the query type: existence, counting, spatial relationship, or comparison.
- **Fusion layer:** concatenation of image features, text features, and question-type features, followed by a feed-forward layer.
- **Dual prediction heads:**
  - a yes/no classification head for existence, spatial, and comparison questions;
  - a count classification head for counting questions.

The custom model is trained with random clean/brightness/noise/text-rephrasing augmentation. The best checkpoint is selected using validation accuracy, and final metrics are reported on the held-out test set.

This design is "custom" because it explicitly uses the known task structure of the synthetic benchmark and separates yes/no reasoning from counting.

## Metrics

We report accuracy under the original and perturbed conditions. We also report robustness degradation:

`Accuracy Drop = Original Accuracy - Perturbed Accuracy`

In addition to aggregate accuracy, we show per-question-type accuracy and automatically extracted robustness failures. A robustness failure is defined as a case where the model is correct on the original input but incorrect after a perturbation.

For paired model comparison, we use McNemar's exact test on matched examples. This test compares cases where one model is correct and the other is incorrect on the same image-query pair. We also compute bootstrap 95% confidence intervals for the paired accuracy difference between BLIP and the custom model.

## Visualization

To make the perturbation effects interpretable, the report includes side-by-side examples showing:

- Original Image
- Perturbed Image
- Query
- Ground Truth
- Clean Model Output
- Perturbed Model Output
- Robustness Failure label when applicable

All images are displayed with visible borders so the white synthetic background has a clear boundary in the final report.
