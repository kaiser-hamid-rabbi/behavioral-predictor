"""ONNX Exporter."""

from __future__ import annotations

import os
import torch
import torch.nn as nn


class OnnxExporter:
    """Exports PyTorch model to ONNX format."""
    
    @staticmethod
    def export_model(
        model: nn.Module, 
        output_path: str,
        seq_length: int = 20
    ) -> int:
        """
        Export to ONNX. 
        Returns file size in bytes.
        """
        model.eval()
        
        # Create dummy inputs
        device = next(model.parameters()).device
        
        # event_ids, device_ids, channel_ids
        dummy_event = torch.zeros((1, seq_length), dtype=torch.long, device=device)
        dummy_device = torch.zeros((1, seq_length), dtype=torch.long, device=device)
        dummy_channel = torch.zeros((1, seq_length), dtype=torch.long, device=device)
        
        # Also need a dummy mask [1, seq_length]
        dummy_mask = torch.zeros((1, seq_length), dtype=torch.bool, device=device)

        # Tracing requires a tuple of arguments matching forward()
        inputs = (dummy_event, dummy_device, dummy_channel, dummy_mask)
        
        torch.onnx.export(
            model,
            inputs,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["event_ids", "device_ids", "channel_ids", "padding_mask"],
            output_names=[
                "p_purchase", "p_churn", "p_next_event", "p_channel",
                "p_engagement", "p_inactivity", "p_action", "p_active_period"
            ],
            dynamic_axes={
                "event_ids": {0: "batch_size", 1: "seq_len"},
                "device_ids": {0: "batch_size", 1: "seq_len"},
                "channel_ids": {0: "batch_size", 1: "seq_len"},
                "padding_mask": {0: "batch_size", 1: "seq_len"},
                "p_purchase": {0: "batch_size"},
                "p_churn": {0: "batch_size"},
                "p_next_event": {0: "batch_size"},
                "p_channel": {0: "batch_size"},
                "p_engagement": {0: "batch_size"},
                "p_inactivity": {0: "batch_size"},
                "p_action": {0: "batch_size"},
                "p_active_period": {0: "batch_size"},
            }
        )
        
        file_size = os.path.getsize(output_path)
        return file_size
