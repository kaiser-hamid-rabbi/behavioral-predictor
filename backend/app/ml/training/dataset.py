"""PyTorch Dataset implementation."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
import pandas as pd

class BehavioralDataset(Dataset):
    """
    Dataset wrapping extracted sequence features and their targets.
    Expects pre-processed data records representing sliding windows.
    """
    def __init__(self, records: list[dict]):
        """
        records: List of dictionaries. Each dict contains:
        {
           'event_name': [list of ints],
           'device_os': [list of ints],
           'channel': [list of ints],
           'padding_mask': [list of bools/ints],
           
           'target_purchase': float,
           'target_churn': float,
           'target_next_event': int,
           'target_channel': int,
           'target_engagement': float,
           'target_inactivity': float,
           'target_recommended_action': int,
           'target_active_period': int,
        }
        """
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        
        return {
            # Inputs
            "event_name": torch.tensor(rec["event_name"], dtype=torch.long),
            "device_os": torch.tensor(rec["device_os"], dtype=torch.long),
            "channel": torch.tensor(rec["channel"], dtype=torch.long),
            "padding_mask": torch.tensor(rec["padding_mask"], dtype=torch.bool),
            
            # Targets
            "target_purchase": torch.tensor([rec.get("target_purchase", 0.0)], dtype=torch.float32),
            "target_churn": torch.tensor([rec.get("target_churn", 0.0)], dtype=torch.float32),
            "target_next_event": torch.tensor(rec.get("target_next_event", 0), dtype=torch.long),
            "target_channel": torch.tensor(rec.get("target_channel", 0), dtype=torch.long),
            "target_engagement": torch.tensor([rec.get("target_engagement", 0.0)], dtype=torch.float32),
            "target_inactivity": torch.tensor([rec.get("target_inactivity", 0.0)], dtype=torch.float32),
            "target_recommended_action": torch.tensor(rec.get("target_recommended_action", 0), dtype=torch.long),
            "target_active_period": torch.tensor(rec.get("target_active_period", 0), dtype=torch.long),
        }
