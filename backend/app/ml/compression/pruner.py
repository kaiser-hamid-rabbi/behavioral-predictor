"""Pruning module."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.utils.prune as prune


class Pruner:
    """Applies unstructured magnitude pruning to model weights."""
    
    @staticmethod
    def apply_pruning(model: nn.Module, amount: float = 0.3) -> nn.Module:
        """
        Prunes the weakest connections in linear and embedding layers.
        Amount parameter determines the percentage to prune (e.g., 0.3 = 30%).
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=amount)
                # Make the pruning permanent
                prune.remove(module, 'weight')
            elif isinstance(module, nn.Embedding):
                prune.l1_unstructured(module, name='weight', amount=amount)
                prune.remove(module, 'weight')
                
        return model
