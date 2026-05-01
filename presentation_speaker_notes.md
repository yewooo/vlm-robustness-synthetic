# 5-Minute Presentation Speaker Notes

## Slide 1: Motivation
Vision-language models can achieve high benchmark accuracy, but this does not necessarily mean they reason reliably. Our project asks whether a model keeps the same answer when the image or the wording changes slightly but the semantic answer should not change. We focus on attribute binding and counting, because these are simple but important multimodal reasoning skills.

## Slide 2: Problem Setup
We use synthetic scenes with colored geometric shapes so that we have exact ground truth and full control over the image. Each sample has an image, a query, and a ground-truth answer. We evaluate four query types: existence, counting, spatial relationship, and comparison. We test original inputs and three perturbations: brightness, noise, and text rephrasing.

## Slide 3: Model
We compare a pretrained BLIP VQA baseline with a custom task-specific model. BLIP uses the Salesforce/blip-vqa-base checkpoint and is evaluated without fine-tuning. The custom model uses a ResNet-18 image encoder, bidirectional GRU text encoder, query-type embedding, and two output heads: one for yes/no questions and one for counting.

## Slide 4: Experiments
Every model is evaluated on the same 600 held-out test examples under original and perturbed conditions. We report exact-match accuracy, robustness degradation, per-task breakdowns, qualitative failure cases, and paired McNemar statistical tests comparing BLIP and the custom model on the same examples.

## Slide 5: Results
BLIP has slightly higher clean accuracy, 0.593 compared with 0.572 for the custom model. However, the custom model is competitive and slightly higher under noise, 0.577 compared with 0.570. The per-task analysis shows that BLIP is stronger on existence and comparison, while the custom model improves counting. The paired statistical tests show that the overall differences are not significant at the 0.05 level.
