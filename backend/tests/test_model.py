import torch
from app.ml.training.model import BehavioralPredictor, NUM_ENGINEERED_FEATURES

def test_model_forward():
    model = BehavioralPredictor(
        vocab_sizes={"event_name": 10, "device_os": 5, "channel": 3, "category": 8, "traffic_source": 6},
        vocab_offsets={},
        d_model=32, nhead=2, num_layers=1, dim_ff=64
    )
    
    batch_size, seq_len = 2, 5
    
    event_ids    = torch.randint(0, 10, (batch_size, seq_len))
    device_ids   = torch.randint(0, 5, (batch_size, seq_len))
    channel_ids  = torch.randint(0, 3, (batch_size, seq_len))
    category_ids = torch.randint(0, 8, (batch_size, seq_len))
    hour_ids     = torch.randint(0, 25, (batch_size, seq_len))
    traffic_ids  = torch.randint(0, 6, (batch_size, seq_len))
    numeric_feats = torch.randn(batch_size, NUM_ENGINEERED_FEATURES)
    padding_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    
    preds = model(event_ids, device_ids, channel_ids, category_ids, hour_ids, traffic_ids, numeric_feats, padding_mask)
    
    assert len(preds) == 8
    assert preds[0].shape == (batch_size, 1)   # purchase
    assert preds[2].shape == (batch_size, 10)  # next_event
    assert preds[7].shape == (batch_size, 4)   # active_period
