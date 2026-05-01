"""BLIP VQA wrapper with explicit model-version logging."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class BLIPConfig:
    model_id: str = "Salesforce/blip-vqa-base"
    display_name: str = "BLIP VQA Base"
    max_new_tokens: int = 5


class BLIPVQAWrapper:
    """Thin wrapper around Hugging Face BLIP VQA.

    The wrapper records the exact model id used in the experiment log, which
    addresses ambiguity such as "BLIP" vs a specific checkpoint.
    """

    def __init__(self, config: BLIPConfig | None = None, device: str | None = None):
        try:
            from transformers import BlipForQuestionAnswering, BlipProcessor
        except ImportError as exc:
            raise ImportError("Install transformers before using BLIPVQAWrapper.") from exc

        self.config = config or BLIPConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained(self.config.model_id)
        self.model = BlipForQuestionAnswering.from_pretrained(self.config.model_id).to(self.device)
        self.model.eval()

    @property
    def name(self) -> str:
        return self.config.display_name

    def predict(self, image, query: str) -> str:
        inputs = self.processor(images=image, text=query, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=self.config.max_new_tokens)
        return self.processor.decode(outputs[0], skip_special_tokens=True).strip()

    def experiment_config(self) -> dict[str, Any]:
        data = asdict(self.config)
        data["device"] = self.device
        data["torch_version"] = torch.__version__
        return data
