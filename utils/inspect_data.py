
import pandas as pd
import os

def check_data(lb5_path, lb50_path):
    print(f"Loading 5lb: {lb5_path}")
    try:
        df5 = pd.read_csv(lb5_path, sep='\t')
        df5 = df5[df5['event'] == 'california_wildfires_2018']
        print("5lb Columns:", df5.columns.tolist())
        print("5lb Shape:", df5.shape)
        print("5lb Sample:\n", df5.head(2))
    except Exception as e:
        print(f"Error loading 5lb: {e}")
        return

    print(f"\nLoading 50lb: {lb50_path}")
    try:
        df50 = pd.read_csv(lb50_path, sep='\t')
        df50 = df50[df50['event'] == 'california_wildfires_2018']
        print("50lb Columns:", df50.columns.tolist())
        print("50lb Shape:", df50.shape)
        print("50lb Sample:\n", df50.head(2))
    except Exception as e:
        print(f"Error loading 50lb: {e}")
        return

    # Check subset
    if 'tweet_id' in df5.columns and 'tweet_id' in df50.columns:
        ids5 = set(df5['tweet_id'].astype(str))
        ids50 = set(df50['tweet_id'].astype(str))
        
        intersection = ids5.intersection(ids50)
        missing = ids5 - ids50
        print(f"\nIntersection count: {len(intersection)}")
        print(f"Missing from 50lb (should be 0 for subset): {len(missing)}")
        if len(missing) > 0:
            print(f"Example missing IDs: {list(missing)[:5]}")
    else:
        print("\n'tweet_id' column not found in both.")

    # Check label distribution
    if 'class_label' in df5.columns:
        print("\n5lb Label dist:\n", df5['class_label'].value_counts())
    if 'class_label' in df50.columns:
        print("\n50lb Label dist:\n", df50['class_label'].value_counts())
        
    # Check for empty text
    if 'tweet_text' in df50.columns:
        empty_text = df50[df50['tweet_text'].isna() | (df50['tweet_text'] == '')]
        print(f"\n50lb Empty text count: {len(empty_text)}")

if __name__ == "__main__":
    base_dir = r"d:\Downloads\Git-Stuff\ssl\data\humaid\anh_4o\sep"
    lb5 = os.path.join(base_dir, "5lb", "1", "labeled.tsv")
    lb50 = os.path.join(base_dir, "50lb", "1", "labeled.tsv")
    check_data(lb5, lb50)
