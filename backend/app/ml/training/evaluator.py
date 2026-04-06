"""Model evaluation across all prediction heads."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List

class MultiTaskEvaluator:
    """Evaluates multi-task model performance."""
    
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        
    def evaluate(self, dataloader: DataLoader) -> dict:
        """Compute metrics per task."""
        self.model.eval()
        
        # Simple tracking of correct predictions / totals
        correct_next_event = 0
        total_samples = 0
        
        preds_pur = []
        targs_pur = []
        
        preds_churn = []
        targs_churn = []
        
        with torch.no_grad():
            for batch in dataloader:
                event_ids = batch["event_name"].to(self.device)
                device_ids = batch["device_os"].to(self.device)
                channel_ids = batch["channel"].to(self.device)
                padding_mask = batch["padding_mask"].to(self.device)
                
                preds = self.model(event_ids, device_ids, channel_ids, padding_mask)
                
                p_pur, p_churn, p_next, p_chan, p_eng, p_inact, p_act, p_per = preds
                
                # Next event accuracy
                pred_next_classes = torch.argmax(p_next, dim=-1)
                targ_next = batch["target_next_event"].to(self.device)
                correct_next_event += (pred_next_classes == targ_next).sum().item()
                total_samples += targ_next.size(0)
                
                # Collect probabilities for ROC/AUC later if needed
                preds_pur.extend(torch.sigmoid(p_pur).cpu().numpy().tolist())
                targs_pur.extend(batch["target_purchase"].cpu().numpy().tolist())
                
                preds_churn.extend(torch.sigmoid(p_churn).cpu().numpy().tolist())
                targs_churn.extend(batch["target_churn"].cpu().numpy().tolist())
                
        metrics = {
            "next_event_accuracy": correct_next_event / max(total_samples, 1),
            # You can add sklearn roc_auc_score here if you want full metrics
            # "purchase_auc": roc_auc_score(targs_pur, preds_pur),
        }
        
        return metrics
