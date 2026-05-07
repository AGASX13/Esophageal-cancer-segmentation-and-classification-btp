"""Module 2: Phikon encoder builder."""

from __future__ import annotations

import torch
from transformers import ViTModel


def build_pathology_encoder(device: str | torch.device = "cuda") -> ViTModel:
    """Load Phikon and prepare it for inference."""
    model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
    model = model.to(device)
    model.eval()
    return model
