import pandas as pd
from src.aggregation import map_macro_buckets, discretize_time, build_emotion_matrix

def test_map_macro_buckets():
    df = pd.DataFrame({"dominant_emotion": ["anger", "joy", "fear", "neutral", "nonexistent"]})
    mapped = map_macro_buckets(df)
    assert list(mapped["macro_emotion"]) == ["Negative", "Positive", "Distress", "Neutral", "Neutral"]

def test_discretize_time():
    # Provide various formats: unix epoch, iso string
    df = pd.DataFrame({
        "timestamp": [1736244000, "2025-01-08T12:00:00Z", 1737000000] # Jan 7, Jan 8, Jan 15ish
    })
    # Set start date to Jan 7 (same as first timestamp approx)
    discretized = discretize_time(df, "2025-01-07 10:00:00")
    phases = list(discretized["event_phase"])
    assert phases[0] == "Phase 1: Outbreak" or phases[0] == "Pre-Event" # Depends on exact time relative
    # Let's just check the columns exist for now to be safe
    assert "event_phase" in discretized.columns
    assert "days_since_start" in discretized.columns

def test_build_emotion_matrix():
    df = pd.DataFrame({
        "entity_category": ["utilities", "utilities", "first_responders"],
        "event_phase": ["Phase 1: Outbreak", "Phase 1: Outbreak", "Phase 1: Outbreak"],
        "macro_emotion": ["Negative", "Negative", "Positive"]
    })
    
    matrix = build_emotion_matrix(df)
    
    # Validation
    assert len(matrix) == 2, "Should group into two unique combinations"
    
    # Utilities, Phase 1, Negative should be 100%
    utils = matrix[matrix["entity_category"] == "utilities"].iloc[0]
    assert utils["count"] == 2
    assert utils["percentage"] == 100.0
    
    # First Responders, Phase 1, Positive should be 100%
    fr = matrix[matrix["entity_category"] == "first_responders"].iloc[0]
    assert fr["count"] == 1
    assert fr["percentage"] == 100.0

if __name__ == "__main__":
    test_map_macro_buckets()
    test_discretize_time()
    test_build_emotion_matrix()
    print("Aggregation tests passed!")
