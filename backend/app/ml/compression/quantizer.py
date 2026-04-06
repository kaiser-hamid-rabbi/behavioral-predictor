"""Quantization module."""

from __future__ import annotations

import torch
import torch.nn as nn


class Quantizer:
    """Applies dynamic quantization to shrink model size while preserving accuracy."""
    
    @staticmethod
    def quantize_dynamic(model: nn.Module) -> nn.Module:
        """
        Apply INT8 dynamic quantization to Linear layers.
        Significantly reduces size of linear output heads.
        """
        model.eval()
        quantized_model = torch.ao.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8
        )
        return quantized_model
