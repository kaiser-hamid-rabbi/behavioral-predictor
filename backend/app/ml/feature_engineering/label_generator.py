"""Label generation for multi-task training."""

from __future__ import annotations

import pandas as pd
from datetime import timedelta

class LabelGenerator:
    """Generates ground-truth labels for sliding window training sequences."""
    
    @staticmethod
    def generate_labels(historical_events: pd.DataFrame, future_events: pd.DataFrame, vocab) -> dict[str, float | int]:
        """
        Given the historical window and the 'future' window of events,
        compute the targets for the multi-head predictor.
        """
        # If no future, we can't train effectively on this sample. 
        # But we handle defaults for robustness.
        
        # 1. Purchase probability target
        purchased = 1.0 if not future_events.empty and "purchase" in future_events["event_name"].values else 0.0
        
        # 2. Churn target
        # Simplified definition: no events for 7 days after the historical window
        if not historical_events.empty and future_events.empty:
            # We don't really know unless we know we are at the end of the total timeline 
            # In a real batch, you'd check diff against global max date. 
            # For sliding window inside user's timeline, if future is missing, it's churn.
            churned = 1.0
        else:
            churned = 0.0

        # 3. Next event target
        next_event = 0 # <PAD>
        if not future_events.empty:
             next_ev_name = future_events.iloc[0]["event_name"]
             next_event = vocab.encode("event_name", next_ev_name)
             
        # 4. Preferred channel (mode in future)
        channel = 0
        if not future_events.empty and not future_events["channel"].mode().empty:
             channel = vocab.encode("channel", future_events["channel"].mode().iloc[0])

        # 5. Engagement Score (normalized count in future window, simple version)
        engagement = float(len(future_events))

        # 6. Inactivity risk (days to next activity)
        inactivity = 0.0
        if not future_events.empty and not historical_events.empty:
            last_hist = pd.to_datetime(historical_events["event_time"].iloc[-1])
            first_fut = pd.to_datetime(future_events["event_time"].iloc[0])
            inactivity = float((first_fut - last_hist).days)
            
        # 7. Recommended Action
        # (Could be the modal category or just a synthesized target)
        # Mocking to 0 for simplicity.
        recommended_action = 0

        # 8. Active Period Segment (Simple hour categorization of next event)
        active_period = 0
        if not future_events.empty:
             hour = pd.to_datetime(future_events.iloc[0]["event_time"]).hour
             if hour < 6: active_period = 0 # Night
             elif hour < 12: active_period = 1 # Morning
             elif hour < 18: active_period = 2 # Afternoon
             else: active_period = 3 # Evening

        return {
            "target_purchase": purchased,
            "target_churn": churned,
            "target_next_event": next_event,
            "target_channel": channel,
            "target_engagement": engagement,
            "target_inactivity": inactivity,
            "target_recommended_action": recommended_action,
            "target_active_period": active_period,
        }
