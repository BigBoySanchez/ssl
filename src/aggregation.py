import pandas as pd
import numpy as np

MACRO_BUCKETS = {
    "anger": "Negative", "annoyance": "Negative", "disapproval": "Negative", "disgust": "Negative",
    "gratitude": "Positive", "admiration": "Positive", "approval": "Positive", "joy": "Positive", "love": "Positive", "optimism": "Positive", "pride": "Positive", "relief": "Positive",
    "fear": "Distress", "nervousness": "Distress", "sadness": "Distress", "grief": "Distress", "remorse": "Distress", "disappointment": "Distress", "embarrassment": "Distress",
    "neutral": "Neutral", "confusion": "Neutral", "curiosity": "Neutral", "realization": "Neutral", "surprise": "Neutral", "amusement": "Positive", "caring": "Positive", "desire": "Positive"
}

def map_macro_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Map the 28 fine-grained emotions into macro-buckets."""
    df = df.copy()
    if "dominant_emotion" not in df.columns:
        raise ValueError("Missing 'dominant_emotion' column")
        
    df["macro_emotion"] = df["dominant_emotion"].map(MACRO_BUCKETS)
    # Handle any missing cleanly (should not happen if all GoEmotions are mapped)
    df["macro_emotion"] = df["macro_emotion"].fillna("Neutral")
    return df

def discretize_time(df: pd.DataFrame, start_date_str: str) -> pd.DataFrame:
    """
    Discretize the timestamps into specific event phases relative to an arbitrary start date.
    Args:
        df: DataFrame with a 'timestamp' column containing Unix epochs or ISO strings.
        start_date_str: The zero-hour start date, e.g., '2025-01-07 10:00:00'.
    """
    df = df.copy()
    
    # Convert timestamps to proper datetime, enforcing UTC directly to avoid mixed timezone errors
    df["datetime"] = pd.to_datetime(df["timestamp"], unit='s', errors='coerce', utc=True)
    # If there are ISO strings, coerce above returns NaT. Try again with iso format where NaT.
    iso_mask = df["datetime"].isna() & df["timestamp"].notna()
    if iso_mask.any():
        df.loc[iso_mask, "datetime"] = pd.to_datetime(df.loc[iso_mask, "timestamp"], errors='coerce', utc=True)
    
    # Ensure UTC 
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize('UTC')
    else:
        df["datetime"] = df["datetime"].dt.tz_convert('UTC')
        
    start_date = pd.to_datetime(start_date_str).tz_localize('UTC')
    
    df["days_since_start"] = (df["datetime"] - start_date).dt.total_seconds() / (24 * 3600)
    
    # Phase buckets: Day 0-2 (Phase 1), Day 3-7 (Phase 2), Day 8+ (Phase 3)
    conditions = [
        (df["days_since_start"] < 0),
        (df["days_since_start"] >= 0) & (df["days_since_start"] <= 2),
        (df["days_since_start"] > 2) & (df["days_since_start"] <= 7),
        (df["days_since_start"] > 7)
    ]
    choices = ["Pre-Event", "Phase 1: Outbreak", "Phase 2: Disruption", "Phase 3: Recovery"]
    
    df["event_phase"] = np.select(conditions, choices, default="Unknown")
    return df

def build_emotion_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs the Emotion x Entity matrix aggregating percentages across daily time bins.
    """
    # Group by Entity, Phase, Macro Emotion
    grouped = df.groupby(["entity_category", "event_phase", "macro_emotion"]).size().reset_index(name="count")
    
    # Calculate totals per entity per phase to get percentages
    totals = grouped.groupby(["entity_category", "event_phase"])["count"].transform("sum")
    grouped["percentage"] = (grouped["count"] / totals) * 100
    
    # Assert Macro-Bucket Sum == Total Processed Rows for safety
    assert grouped["count"].sum() == len(df), "Row loss detected during aggregation! Matrix sums do not match input."
    
    return grouped
