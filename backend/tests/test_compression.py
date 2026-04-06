import tempfile
import os
from app.ml.training.model import BehavioralPredictor
from app.ml.compression.exporter import OnnxExporter

def test_onnx_export():
    model = BehavioralPredictor(
        vocab_sizes={"event_name": 10, "device_os": 2, "channel": 2},
        vocab_offsets={},
        d_model=16,
        nhead=2,
        num_layers=1,
        dim_ff=32
    )
    
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        export_path = f.name
        
    try:
        size = OnnxExporter.export_model(model, export_path, seq_length=5)
        # Verify size < 1MB
        assert size < 1024 * 1024
        assert os.path.exists(export_path)
    finally:
        if os.path.exists(export_path):
            os.remove(export_path)
