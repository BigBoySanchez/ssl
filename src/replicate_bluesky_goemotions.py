import argparse
import pandas as pd
from transformers import pipeline
import torch
import os
import sys

# Ensure we can import from src directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ingestion import ingest_reddit_data, ingest_bluesky_data

def run_goemotions_on_raw_data(reddit_subs, reddit_comms, bluesky_jsonl, output_csv):
    df_list = []
    
    # Ingest Reddit Datasets
    if reddit_subs and reddit_comms:
        print("Loading Reddit data...")
        try:
            reddit_df = ingest_reddit_data(reddit_subs, reddit_comms)
            df_list.append(reddit_df)
            print(f"Loaded {len(reddit_df)} Reddit posts/comments.")
        except Exception as e:
            print(f"Failed to load Reddit data: {e}")
    elif reddit_subs or reddit_comms:
        print("Warning: Both submissions and comments files are required for Reddit data. Skipping Reddit.")
        
    # Ingest Bluesky Dataset
    if bluesky_jsonl:
        print(f"Loading Bluesky data from {bluesky_jsonl}...")
        try:
            bluesky_df = ingest_bluesky_data(bluesky_jsonl)
            df_list.append(bluesky_df)
            print(f"Loaded {len(bluesky_df)} Bluesky posts.")
        except Exception as e:
            print(f"Failed to load Bluesky data: {e}")
            
    if not df_list:
        print("Error: No valid input files were successfully loaded.")
        return
        
    # Combine datasets
    df = pd.concat(df_list, ignore_index=True)
    
    # Basic text cleaning: drop empty strings or nulls to prevent pipeline errors
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.strip() != '']
    
    print(f"\nTotal posts collected across all platforms: {len(df)}")
    
    print("\nLoading SamLowe/roberta-base-go_emotions model...")
    if torch.cuda.is_available():
        device = 0
        print("Using CUDA GPU")
    else:
        device = -1
        print("Using CPU")
        
    pipe = pipeline("text-classification", 
                    model="SamLowe/roberta-base-go_emotions", 
                    top_k=None, # Extract all 28 GOEmotions emotion scores
                    truncation=True, 
                    max_length=512,
                    device=device)

    texts = df["text"].astype(str).tolist()
    
    print(f"\nRunning inference ...")
    results = []
    batch_size = 64 # Recommended for a 8GB/16GB GPU avoiding CUDA OOM
    
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        out = pipe(batch)
        if len(out) == 0:
            continue
            
        if isinstance(out[0], list):
            results.extend(out)
        else:
            results.append(out)
            
        current_batch = i // batch_size + 1
        # Print progress every 10%
        if current_batch % max(1, (total_batches // 10)) == 0:
            print(f"Processed {current_batch}/{total_batches} batches ({(current_batch/total_batches)*100:.0f}%)")
        
    print("\nMapping predictions to Macro-Buckets (VADER Equivalent)...")
    
    macro_mapping = {
        "anger": "neg", "annoyance": "neg", "disapproval": "neg", "disgust": "neg",
        "gratitude": "pos", "admiration": "pos", "approval": "pos", "joy": "pos", 
        "love": "pos", "optimism": "pos", "pride": "pos", "relief": "pos", "amusement": "pos", "caring": "pos", "desire": "pos",
        "fear": "distress", "nervousness": "distress", "sadness": "distress", "grief": "distress", 
        "remorse": "distress", "disappointment": "distress", "embarrassment": "distress",
        "neutral": "neu", "confusion": "neu", "curiosity": "neu", "realization": "neu", "surprise": "neu"
    }
    
    goemo_neg, goemo_pos, goemo_neu, goemo_distress, dominant_emo = [], [], [], [], []
    
    for res in results:
        scores = {"neg": 0.0, "pos": 0.0, "neu": 0.0, "distress": 0.0}
        sorted_res = sorted(res, key=lambda x: x['score'], reverse=True)
        dominant_emo.append(sorted_res[0]['label'])
        
        for item in res:
            label = item['label']
            score = item['score']
            macro = macro_mapping.get(label, "neu")
            scores[macro] += score
            
        goemo_neg.append(scores["neg"])
        goemo_pos.append(scores["pos"])
        goemo_neu.append(scores["neu"])
        goemo_distress.append(scores["distress"])
        
    df["goemo_neg"] = goemo_neg
    df["goemo_pos"] = goemo_pos
    df["goemo_neu"] = goemo_neu
    df["goemo_distress"] = goemo_distress
    df["goemo_dominant"] = dominant_emo
    
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    print(f"\nSaving generated dataset to {output_csv}...")
    df.to_csv(output_csv, index=False)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run goemotions on raw JSON/JSONL datasets")
    parser.add_argument("--bluesky", type=str, help="Path to Bluesky JSONL file", default=None)
    parser.add_argument("--reddit_subs", type=str, help="Path to Reddit Submissions JSON file", default=None)
    parser.add_argument("--reddit_comms", type=str, help="Path to Reddit Comments JSON file", default=None)
    parser.add_argument("--output", "-o", type=str, required=True, help="Output CSV file path")
    
    args = parser.parse_args()
    
    run_goemotions_on_raw_data(args.reddit_subs, args.reddit_comms, args.bluesky, args.output)
