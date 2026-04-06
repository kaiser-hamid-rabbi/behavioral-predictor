import pandas as pd
from app.ml.feature_engineering.feature_builder import FeatureBuilder

def test_build_user_features():
    df = pd.DataFrame([
        {'event_time': '2024-01-01T10:00:00Z', 'event_name': 'viewcontent', 'category': 'books', 'session_id': 's1', 'channel': 'app'},
        {'event_time': '2024-01-01T10:05:00Z', 'event_name': 'purchase', 'category': 'books', 'session_id': 's1', 'channel': 'app'}
    ])
    
    features = FeatureBuilder.build_user_features(df)
    
    assert features["total_events"] == 2
    assert features["purchase_count"] == 1
    assert features["preferred_channel"] == "app"
    assert features["unique_categories"] == 1
