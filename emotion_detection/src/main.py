import pandas as pd
import os
from src.ingestion import ingest_reddit_data, ingest_bluesky_data, preprocess_text, split_and_filter_sentences
from src.inference import run_inference
from src.aggregation import map_macro_buckets, discretize_time, build_emotion_matrix

def main():
    # Define entities based on the instructions
    entities = {
        "emergency_management": ["FEMA", "LA County", "emergency management", "evacuation order", "sheriff"],
        "first_responders": ["firefighter", "paramedic", "first responder", "CAL FIRE", "LAFD"],
        "utilities": ["PG&E", "Southern California Edison", "SCE", "Edison", "power company", "utility", "PSPS"]
    }
    
    print("Phase 1: Ingesting Data...")
    reddit_df = ingest_reddit_data(
        "data/emotion_detection/palisades_submissions_reddit.json",
        "data/emotion_detection/palisades_comments_reddit.json"
    )
    bluesky_df = ingest_bluesky_data("data/emotion_detection/palisades-fire-bluesky.jsonl")
    
    # Combine datasets
    df = pd.concat([reddit_df, bluesky_df], ignore_index=True)
    print(f"Total raw posts ingested: {len(df)}")
    
    print("Phase 1: Preprocessing and Filtering...")
    processed_rows = []
    
    # Normally we would use df.apply, but we are expanding 1 post -> multiple sentences
    for _, row in df.iterrows():
        text = preprocess_text(row['text'])
        sentences = split_and_filter_sentences(text, entities)
        
        for sent in sentences:
            processed_rows.append({
                "id": row["id"],
                "platform": row["platform"],
                "timestamp": row["timestamp"],
                "post_type": row["post_type"],
                "text": sent["text"],
                "entity_category": sent["entity_category"]
            })
            
    filtered_df = pd.DataFrame(processed_rows)
    print(f"Total entity-matched sentences: {len(filtered_df)}")
    
    if len(filtered_df) == 0:
        print("No sentences matched entities. Exiting.")
        return
        
    print("Phase 2: Running Inference using roberta-base-goemotions...")
    
    inferred_df = run_inference(filtered_df, mock=False)
    
    print("Phase 3: Aggregation...")
    # Map to macro buckets
    macro_df = map_macro_buckets(inferred_df)
    
    # LA Wildfire zero-hour approx Jan 7, 2025
    time_df = discretize_time(macro_df, start_date_str="2025-01-07 10:00:00")
    
    # Build Matrix
    matrix = build_emotion_matrix(time_df)
    
    print("\nEmotion x Entity Matrix (Percentages):")
    print(matrix.to_string())
    
    print("\nPhase 4: Validation Setup...")
    # Prepare a stratified sample for human annotators
    # We group by platform and entity_category to ensure steady distribution
    # If sample_df is large enough, we can stratify. Since we only ran 100, we just save all.
    validation_sample = time_df[["id", "platform", "post_type", "entity_category", "text", "dominant_emotion", "macro_emotion"]]
    
    os.makedirs("output", exist_ok=True)
    validation_sample.to_csv("output/validation_sample.csv", index=False)
    matrix.to_csv("output/emotion_matrix.csv", index=False)
    print("Saved pipeline outputs to 'output/' folder.")
    
if __name__ == "__main__":
    main()
