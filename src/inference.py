import pandas as pd

def run_inference(df: pd.DataFrame, mock=False) -> pd.DataFrame:
    """
    Run the HuggingFace pipeline on the text column to extract dominant emotion.
    Args:
        df: DataFrame containing a 'text' column.
        mock: If True, returns dummy predictions to skip loading the real model (useful for fast tests).
    Returns:
        DataFrame with 'dominant_emotion' and 'emotion_score' columns added.
    """
    if "text" not in df.columns:
        raise ValueError("DataFrame must contain a 'text' column.")
        
    df = df.copy()
    
    if mock:
        # Mocking the pipeline output
        import random
        emotions = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]
        
        df["dominant_emotion"] = [random.choice(emotions) for _ in range(len(df))]
        df["emotion_score"] = [random.uniform(0.5, 0.99) for _ in range(len(df))]
        return df

    # Actual pipeline initialization
    from transformers import pipeline
    
    # Initialize the model that outputs all 28 classes
    pipe = pipeline("text-classification", 
                    model="SamLowe/roberta-base-go_emotions", # SamLowe model is a popular fine-tune of roberta for goemotions
                    top_k=1, # We only need the top prediction (argmax)
                    truncation=True, 
                    max_length=512,
                    device=0)
    
    # Run predictions in batch to avoid CUDA OOM
    texts = df["text"].tolist()
    
    results = []
    batch_size = 64 # Adjust if you still get OOM errors
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        results.extend(pipe(batch))
        
    # results is a list of lists because top_k returns a list of dicts for each input
    # e.g., [[{'label': 'anger', 'score': 0.9}]]
    dominants = []
    scores = []
    
    for res in results:
        # Get the top prediction
        top_pred = res[0] 
        dominants.append(top_pred["label"])
        scores.append(top_pred["score"])
        
    df["dominant_emotion"] = dominants
    df["emotion_score"] = scores
    
    return df
