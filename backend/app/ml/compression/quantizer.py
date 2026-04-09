"""Quantization module."""

from __future__ import annotations

import platform

import torch
import torch.nn as nn


class Quantizer:
    """Applies dynamic quantization to shrink model size while preserving accuracy."""
    
    @staticmethod
    def quantize_dynamic(model: nn.Module) -> nn.Module:
        """
        Apply INT8 dynamic quantization to Linear layers.
        Significantly reduces size of linear output heads.
        
        Automatically selects the correct quantization backend:
        - fbgemm: x86 (Intel/AMD servers, CI pipelines)
        - qnnpack: ARM (Apple Silicon M1/M2/M3, mobile, edge devices)
        """
        model.eval()
        
        # Select quantization engine based on CPU architecture
        arch = platform.machine().lower()
        if arch in ("arm64", "aarch64"):
            torch.backends.quantized.engine = "qnnpack"
        else:
            torch.backends.quantized.engine = "fbgemm"
        
        quantized_model = torch.ao.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8
        )
        return quantized_model

