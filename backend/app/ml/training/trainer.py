"""Training loop implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict
from tqdm import tqdm

class MultiTaskTrainer:
    """Handles the multi-task training loop for BehavioralPredictor."""
    
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        
        # Loss functions for the different heads
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
        # We can dynamically weight the losses if desired
        self.loss_weights = {
            "purchase": 1.0,
            "churn": 1.0,
            "next_event": 0.5,
            "channel": 0.5,
            "engagement": 0.1,
            "inactivity": 0.1,
            "action": 0.2,
            "active_period": 0.2,
        }

    def compute_loss(self, preds: tuple, batch: dict) -> tuple[torch.Tensor, dict]:
        """Compute combined loss across all heads."""
        p_purchase, p_churn, p_next_event, p_channel, p_engagement, p_inactivity, p_action, p_active_period = preds
        
        l_pur = self.bce_loss(p_purchase, batch["target_purchase"].to(self.device))
        l_churn = self.bce_loss(p_churn, batch["target_churn"].to(self.device))
        
        l_next = self.ce_loss(p_next_event, batch["target_next_event"].to(self.device))
        l_chan = self.ce_loss(p_channel, batch["target_channel"].to(self.device))
        
        l_eng = self.mse_loss(p_engagement, batch["target_engagement"].to(self.device))
        l_inact = self.mse_loss(p_inactivity, batch["target_inactivity"].to(self.device))
        
        l_act = self.ce_loss(p_action, batch["target_recommended_action"].to(self.device))
        l_per = self.ce_loss(p_active_period, batch["target_active_period"].to(self.device))
        
        total_loss = (
            self.loss_weights["purchase"] * l_pur +
            self.loss_weights["churn"] * l_churn +
            self.loss_weights["next_event"] * l_next +
            self.loss_weights["channel"] * l_chan +
            self.loss_weights["engagement"] * l_eng +
            self.loss_weights["inactivity"] * l_inact +
            self.loss_weights["action"] * l_act +
            self.loss_weights["active_period"] * l_per
        )
        
        metrics = {
            "loss_pur": l_pur.item(),
            "loss_churn": l_churn.item(),
            "loss_next": l_next.item(),
            "loss_total": total_loss.item()
        }
        
        return total_loss, metrics

    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer) -> dict:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch in tqdm(dataloader, desc="Training"):
            optimizer.zero_grad()
            
            event_ids = batch["event_name"].to(self.device)
            device_ids = batch["device_os"].to(self.device)
            channel_ids = batch["channel"].to(self.device)
            padding_mask = batch["padding_mask"].to(self.device)
            
            preds = self.model(event_ids, device_ids, channel_ids, padding_mask)
            
            loss, metrics = self.compute_loss(preds, batch)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
        return {"train_loss": total_loss / len(dataloader)}
    
    def validate(self, dataloader: DataLoader) -> dict:
        """Run validation loop."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                event_ids = batch["event_name"].to(self.device)
                device_ids = batch["device_os"].to(self.device)
                channel_ids = batch["channel"].to(self.device)
                padding_mask = batch["padding_mask"].to(self.device)
                
                preds = self.model(event_ids, device_ids, channel_ids, padding_mask)
                loss, _ = self.compute_loss(preds, batch)
                total_loss += loss.item()
                
        return {"val_loss": total_loss / len(dataloader)}
