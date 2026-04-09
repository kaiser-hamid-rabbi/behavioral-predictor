"""ONNX Exporter."""

from __future__ import annotations

import onnx
import logging
import os
import warnings

import torch
import torch.nn as nn

from app.ml.training.model import NUM_ENGINEERED_FEATURES

logger = logging.getLogger(__name__)


class OnnxExporter:
    """Exports PyTorch model to ONNX format."""
    
    @staticmethod
    def export_model(
        model: nn.Module, 
        output_path: str,
        seq_length: int = 20
    ) -> int:
        """Export to ONNX. Returns file size in bytes."""
        model.eval()
        device = next(model.parameters()).device
        
        # Categorical inputs [batch, seq]
        dummy_event    = torch.zeros((1, seq_length), dtype=torch.long, device=device)
        dummy_device   = torch.zeros((1, seq_length), dtype=torch.long, device=device)
        dummy_channel  = torch.zeros((1, seq_length), dtype=torch.long, device=device)
        dummy_category = torch.zeros((1, seq_length), dtype=torch.long, device=device)
        dummy_hour     = torch.zeros((1, seq_length), dtype=torch.long, device=device)
        dummy_traffic  = torch.zeros((1, seq_length), dtype=torch.long, device=device)
        # Numeric features [batch, N]
        dummy_numeric  = torch.zeros((1, NUM_ENGINEERED_FEATURES), dtype=torch.float32, device=device)
        # Mask [batch, seq]
        dummy_mask     = torch.zeros((1, seq_length), dtype=torch.bool, device=device)

        inputs = (
            dummy_event, dummy_device, dummy_channel,
            dummy_category, dummy_hour, dummy_traffic,
            dummy_numeric, dummy_mask,
        )
        
        # Single self-contained file export [Mandatory for Browser Runtime]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*torchvision.*")
            
            torch.onnx.export(
                model,
                inputs,
                output_path,
                export_params=True,
                opset_version=15,  # Maximum cross-browser stability
                do_constant_folding=True,
                input_names=[
                    "event_ids", "device_ids", "channel_ids",
                    "category_ids", "hour_ids", "traffic_ids",
                    "numeric_features", "padding_mask",
                ],
                output_names=[
                    "p_purchase", "p_churn", "p_next_event", "p_channel",
                    "p_engagement", "p_inactivity", "p_action", "p_active_period"
                ],
                dynamic_axes={
                    "event_ids": {0: "batch", 1: "seq"},
                    "device_ids": {0: "batch", 1: "seq"},
                    "channel_ids": {0: "batch", 1: "seq"},
                    "category_ids": {0: "batch", 1: "seq"},
                    "hour_ids": {0: "batch", 1: "seq"},
                    "traffic_ids": {0: "batch", 1: "seq"},
                    "numeric_features": {0: "batch"},
                    "padding_mask": {0: "batch", 1: "seq"},
                }
            )

        # CRITICAL FIX: Reload and re-save using onnx library to merge external data 
        # into a single self-contained file for browser compatibility.
        temp_model = onnx.load(output_path)
        onnx.save_model(
            temp_model, 
            output_path, 
            save_as_external_data=False
        )
        
        # Cleanup any stray .data files created by the initial torch export
        data_path = output_path + ".data"
        if os.path.exists(data_path):
            os.remove(data_path)
        
        return os.path.getsize(output_path)
