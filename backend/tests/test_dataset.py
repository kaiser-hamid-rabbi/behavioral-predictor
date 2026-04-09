import torch
from app.ml.training.dataset import BehavioralDataset

def test_dataset_getitem():
    records = [{
        "event_name": [1, 2, 3],
        "device_os": [1, 1, 1],
        "channel": [2, 2, 2],
        "category": [3, 3, 3],
        "hour": [10, 11, 12],
        "traffic_source": [1, 2, 1],
        "padding_mask": [False, False, False],
        "numeric_features": [0.1, 0.2, 0.3, 0.5, 0.1, 0.4],
        "target_purchase": 1.0,
        "target_next_event": 4
    }]
    
    dataset = BehavioralDataset(records)
    assert len(dataset) == 1
    
    item = dataset[0]
    assert item["event_name"].shape == (3,)
    assert item["category"].shape == (3,)
    assert item["hour"].shape == (3,)
    assert item["traffic_source"].shape == (3,)
    assert item["numeric_features"].shape == (6,)
    assert item["target_purchase"].item() == 1.0
    assert item["target_next_event"].item() == 4
