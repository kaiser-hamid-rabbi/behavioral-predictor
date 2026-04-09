"""PyTorch Dataset implementation."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class BehavioralDataset(Dataset):
    """
    Dataset wrapping extracted sequence features and their targets.
    Includes both categorical embeddings and engineered numeric features.
    """
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        seq_len = len(rec["event_name"])
        
        return {
            # Categorical inputs (per timestep)
            "event_name":     torch.tensor(rec["event_name"], dtype=torch.long),
            "device_os":      torch.tensor(rec["device_os"], dtype=torch.long),
            "channel":        torch.tensor(rec["channel"], dtype=torch.long),
            "category":       torch.tensor(rec.get("category", [0] * seq_len), dtype=torch.long),
            "hour":           torch.tensor(rec.get("hour", [0] * seq_len), dtype=torch.long),
            "traffic_source": torch.tensor(rec.get("traffic_source", [0] * seq_len), dtype=torch.long),
            "padding_mask":   torch.tensor(rec["padding_mask"], dtype=torch.bool),
            
            # Engineered numeric features (per window)
            "numeric_features": torch.tensor(rec.get("numeric_features", [0.0] * 6), dtype=torch.float32),
            
            # Targets
            "target_purchase":            torch.tensor([rec.get("target_purchase", 0.0)], dtype=torch.float32),
            "target_churn":               torch.tensor([rec.get("target_churn", 0.0)], dtype=torch.float32),
            "target_next_event":          torch.tensor(rec.get("target_next_event", 0), dtype=torch.long),
            "target_channel":             torch.tensor(rec.get("target_channel", 0), dtype=torch.long),
            "target_engagement":          torch.tensor([rec.get("target_engagement", 0.0)], dtype=torch.float32),
            "target_inactivity":          torch.tensor([rec.get("target_inactivity", 0.0)], dtype=torch.float32),
            "target_recommended_action":  torch.tensor(rec.get("target_recommended_action", 0), dtype=torch.long),
            "target_active_period":       torch.tensor(rec.get("target_active_period", 0), dtype=torch.long),
        }
