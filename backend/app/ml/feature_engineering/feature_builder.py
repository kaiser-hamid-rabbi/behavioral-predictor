"""Feature extraction from raw event data."""

from __future__ import annotations

import pandas as pd
from datetime import datetime

class FeatureBuilder:
    """Builds aggregated user features and sequence features from event data."""
    
    @staticmethod
    def build_user_features(user_events: pd.DataFrame) -> dict:
        """
        Compute static and aggregated features for a single user's event history.
        """
        if user_events.empty:
            return {}

        df = user_events.sort_values("event_time")
        
        # Aggregation features
        total_events = len(df)
        purchase_count = len(df[df["event_name"] == "purchase"])
        unique_categories = df["category"].nunique()
        unique_sessions = df["session_id"].nunique()

        # Recency / Time features
        now = datetime.now().astimezone()
        last_event_time = pd.to_datetime(df["event_time"].iloc[-1])
        # Make dummy naive datetime tz-aware if needed
        if last_event_time.tzinfo is None:
             last_event_time = last_event_time.tz_localize('UTC')

        days_since_active = (now - last_event_time).days

        # Preferences
        most_common_channel = df["channel"].mode()[0] if not df["channel"].mode().empty else None
        
        return {
            "total_events": total_events,
            "purchase_count": purchase_count,
            "unique_categories": unique_categories,
            "session_count": unique_sessions,
            "days_since_active": days_since_active,
            "preferred_channel": most_common_channel,
        }

    @staticmethod
    def extract_sequence(user_events: pd.DataFrame, vocab, max_length: int = 20) -> dict[str, list[int]]:
        """
        Convert chronological raw events into numeric sequences for model input.
        Pads or truncates effectively to `max_length`.
        """
        df = user_events.sort_values("event_time").tail(max_length)
        
        # We need padding if length < max_length
        pad_len = max_length - len(df)
        
        seq = {
            "event_name": [0] * pad_len,
            "device_os": [0] * pad_len,
            "channel": [0] * pad_len,
            "traffic_source": [0] * pad_len,
            "category": [0] * pad_len,
        }
        
        for _, row in df.iterrows():
            seq["event_name"].append(vocab.encode("event_name", row["event_name"]))
            seq["device_os"].append(vocab.encode("device_os", row["device_os"]))
            seq["channel"].append(vocab.encode("channel", row["channel"]))
            seq["traffic_source"].append(vocab.encode("traffic_source", row["traffic_source"]))
            seq["category"].append(vocab.encode("category", row["category"]))
            
        return seq
