def test_inference_pipeline_shape():
    # Mock input to ensure we process sentences correctly
    from src.inference import run_inference
    import pandas as pd
    
    # Dummy data
    data = [
        {"id": "1", "text": "FEMA works hard.", "entity_category": "emergency_management"},
        {"id": "2", "text": "PG&E is awful.", "entity_category": "utilities"}
    ]
    df = pd.DataFrame(data)
    
    # In a real test, this would use a tiny mock model or run a fast subset.
    # To keep test fast, our stub run_inference should handle it
    res_df = run_inference(df, mock=True)
    
    # Validate output shape
    assert "dominant_emotion" in res_df.columns, "Must have dominant emotion"
    assert "emotion_score" in res_df.columns, "Should have a confidence score"
    assert len(res_df) == 2, "Must output the same number of rows as input"
    # Basic bounds check on probability if possible
    assert (res_df["emotion_score"] >= 0).all() and (res_df["emotion_score"] <= 1).all()

if __name__ == "__main__":
    test_inference_pipeline_shape()
    print("Inference tests passed!")
