import torch
from app.ml.training.model import BehavioralPredictor

def test_model_forward():
    model = BehavioralPredictor(
        vocab_sizes={"event_name": 10, "device_os": 5, "channel": 3},
        vocab_offsets={},
        d_model=32,
        nhead=2,
        num_layers=1,
        dim_ff=64
    )
    
    batch_size = 2
    seq_len = 5
    
    event_ids = torch.randint(0, 10, (batch_size, seq_len))
    device_ids = torch.randint(0, 5, (batch_size, seq_len))
    channel_ids = torch.randint(0, 3, (batch_size, seq_len))
    padding_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    
    preds = model(event_ids, device_ids, channel_ids, padding_mask)
    
    assert len(preds) == 8
    # Purchase Head Check
    assert preds[0].shape == (batch_size, 1)
    # Next Event Head shape check
    assert preds[2].shape == (batch_size, 10)
