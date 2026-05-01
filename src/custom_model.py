"""Custom dual-head VQA model used for the synthetic-data experiment.

Architecture summary:
- Image encoder: ResNet-18 pretrained on ImageNet, final classifier removed.
- Text encoder: learned token embeddings followed by a bidirectional GRU.
- Query-type embedding: explicit embedding for existence/count/spatial/comparison.
- Fusion: concatenate image, text, and query-type features.
- Output heads:
  1. yes/no head for existence, spatial, and comparison questions.
  2. count head for counting questions.

This is custom because it is task-specific and does not use a pretrained
language decoder; unlike BLIP, it predicts over a small answer space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


QUERY_TYPE_TO_ID = {
    "existence": 0,
    "spatial": 1,
    "comparison": 2,
    "count": 3,
}


@dataclass(frozen=True)
class CustomModelConfig:
    checkpoint_path: str = "custom_dual_head_model.pth"
    text_embed_dim: int = 96
    qtype_embed_dim: int = 32
    hidden_dim: int = 192
    fusion_dim: int = 384
    max_query_len: int = 20
    display_name: str = "Custom Dual-Head VQA Model"


class DualHeadCustomVQAModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        query_type_vocab_size: int = 4,
        text_embed_dim: int = 96,
        qtype_embed_dim: int = 32,
        hidden_dim: int = 192,
        fusion_dim: int = 384,
        num_count_classes: int = 7,
    ):
        super().__init__()
        self.num_count_classes = num_count_classes
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()
        self.embedding = nn.Embedding(vocab_size, text_embed_dim, padding_idx=0)
        self.gru = nn.GRU(text_embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.qtype_embedding = nn.Embedding(query_type_vocab_size, qtype_embed_dim)
        combined_dim = 512 + (hidden_dim * 2) + qtype_embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.yesno_head = nn.Linear(fusion_dim // 2, 2)
        self.count_head = nn.Linear(fusion_dim // 2, num_count_classes)

    def forward(self, images, query_ids, qtype_ids):
        image_features = self.cnn(images)
        embedded = self.embedding(query_ids)
        _, hidden = self.gru(embedded)
        text_features = torch.cat([hidden[-2], hidden[-1]], dim=1)
        qtype_features = self.qtype_embedding(qtype_ids)
        fused = torch.cat([image_features, text_features, qtype_features], dim=1)
        fused = self.fusion(fused)
        return self.yesno_head(fused), self.count_head(fused)


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def encode_query(query: str, vocab: dict[str, int], max_len: int = 20) -> torch.Tensor:
    tokens = query.lower().replace("?", "").replace(",", "").split()
    ids = [vocab.get(token, vocab.get("<unk>", 1)) for token in tokens][:max_len]
    ids += [vocab.get("<pad>", 0)] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


class CustomVQAWrapper:
    """Inference wrapper for the custom dual-head model checkpoint."""

    def __init__(self, config: CustomModelConfig | None = None, device: str | None = None):
        self.config = config or CustomModelConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device)
        self.vocab = checkpoint["vocab"]
        self.num_count_classes = checkpoint["num_count_classes"]
        self.model = DualHeadCustomVQAModel(
            vocab_size=len(self.vocab),
            text_embed_dim=self.config.text_embed_dim,
            qtype_embed_dim=self.config.qtype_embed_dim,
            hidden_dim=self.config.hidden_dim,
            fusion_dim=self.config.fusion_dim,
            num_count_classes=self.num_count_classes,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.transform = build_transform()

    @property
    def name(self) -> str:
        return self.config.display_name

    def predict(self, image: Image.Image, query: str, query_type: str) -> str:
        image_tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        query_ids = encode_query(query, self.vocab, self.config.max_query_len).unsqueeze(0).to(self.device)
        qtype_ids = torch.tensor([QUERY_TYPE_TO_ID[query_type]], dtype=torch.long).to(self.device)
        with torch.no_grad():
            yesno_logits, count_logits = self.model(image_tensor, query_ids, qtype_ids)
        if query_type == "count":
            return str(count_logits.argmax(dim=1).item())
        return "yes" if yesno_logits.argmax(dim=1).item() == 1 else "no"

    def experiment_config(self) -> dict[str, Any]:
        return {
            "display_name": self.config.display_name,
            "checkpoint_path": self.config.checkpoint_path,
            "image_encoder": "ResNet-18 pretrained on ImageNet",
            "text_encoder": "Bidirectional GRU over learned token embeddings",
            "fusion": "concatenation of image, text, and query-type features",
            "heads": ["yes/no classification", "count classification"],
            "device": self.device,
            "torch_version": torch.__version__,
            "vocab_size": len(self.vocab),
            "num_count_classes": self.num_count_classes,
        }
