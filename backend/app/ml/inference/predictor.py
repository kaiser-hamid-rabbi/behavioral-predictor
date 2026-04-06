"""
In-memory predictor using ONNX Runtime.

This module initializes a singleton ONNX session server-side for predictions.
"""

from __future__ import annotations

import onnxruntime as ort
import numpy as np
import uuid
import threading
from typing import Any

from app.schemas.event import EventCreate
from app.ml.feature_engineering.vocabulary import Vocabulary


class PredictorEngine:
    def __init__(self, model_path: str, vocab_dir: str):
        self.session = ort.InferenceSession(model_path)
        self.vocab = Vocabulary()
        self.vocab.load(vocab_dir)

    async def predict(self, user_id: uuid.UUID, recent_events: list[EventCreate]) -> dict[str, Any]:
        """
        Execute prediction using ONNX runtime.
        """
        # Preprocess events
        # Real impl would fetch historical events for user_id and append recent_events
        # Then create sliding window. For now we use recent_events directly.
        max_length = 20
        events = recent_events[-max_length:]
        pad_len = max_length - len(events)
        
        event_ids = [0] * pad_len
        device_ids = [0] * pad_len
        channel_ids = [0] * pad_len
        padding_mask = [True] * pad_len
        
        for e in events:
            event_ids.append(self.vocab.encode('event_name', e.event_name))
            device_ids.append(self.vocab.encode('device_os', e.device_os))
            channel_ids.append(self.vocab.encode('channel', e.channel))
            padding_mask.append(False)
            
        # Convert to numpy explicitly as np.int64 and np.bool_ exactly matches standard ONNX export for PyTorch Long/Bool
        inputs = {
            "event_ids": np.array([event_ids], dtype=np.int64),
            "device_ids": np.array([device_ids], dtype=np.int64),
            "channel_ids": np.array([channel_ids], dtype=np.int64),
            "padding_mask": np.array([padding_mask], dtype=bool)
        }
        
        # Run inference
        outputs = self.session.run(None, inputs)
        
        # Unpack outputs based on our exporter names
        # "p_purchase", "p_churn", "p_next_event", "p_channel",
        # "p_engagement", "p_inactivity", "p_action", "p_active_period"
        
        # Sigmoid for binary
        def sigmoid(x: float) -> float:
            import math
            return 1 / (1 + math.exp(-x))
            
        p_purchase = sigmoid(float(outputs[0][0][0]))
        p_churn = sigmoid(float(outputs[1][0][0]))
        
        # Argmax for categorical
        next_event_id = int(np.argmax(outputs[2][0]))
        channel_id = int(np.argmax(outputs[3][0]))
        
        # Reverse lookup for categorical
        next_event = "unknown"
        preferred_channel = "unknown"
        
        for name, idx in self.vocab.mappings.get("event_name", {}).items():
            if idx == next_event_id: next_event = name
            
        for name, idx in self.vocab.mappings.get("channel", {}).items():
            if idx == channel_id: preferred_channel = name
            
        # Regression
        engagement = float(outputs[4][0][0])
        inactivity = float(outputs[5][0][0])
        
        # Mocking actions/periods
        recommended_action = f"action_{np.argmax(outputs[6][0])}"
        active_time = f"period_{np.argmax(outputs[7][0])}"
        
        return {
            "purchase_probability": p_purchase,
            "churn_risk": p_churn,
            "next_event": next_event,
            "preferred_channel": preferred_channel,
            "engagement_score": engagement,
            "inactivity_risk": inactivity,
            "recommended_action": recommended_action,
            "active_time": active_time
        }


# Singleton pattern
_predictor_instance = None
_lock = threading.Lock()

def initialize_predictor(model_path: str, vocab_dir: str) -> None:
    global _predictor_instance
    with _lock:
        if _predictor_instance is None:
            _predictor_instance = PredictorEngine(model_path, vocab_dir)

def get_predictor() -> PredictorEngine | None:
    return _predictor_instance
