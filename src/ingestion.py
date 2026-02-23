import json
import pandas as pd
import re

def ingest_reddit_data(submissions_file, comments_file):
    with open(submissions_file, 'r') as f:
        subs = json.load(f)
        
    with open(comments_file, 'r') as f:
        comms = json.load(f)
        
    data = []
    
    for sub in subs:
        text = f"{sub.get('title', '')} {sub.get('selftext', '')}".strip()
        data.append({
            'id': sub.get('id'),
            'parent_id': None,
            'post_type': 'submission',
            'timestamp': sub.get('created_utc'),
            'text': text,
            'platform': 'reddit'
        })
        
    for comm in comms:
        data.append({
            'id': comm.get('id'),
            'parent_id': comm.get('parent_id'),
            'post_type': 'comment',
            'timestamp': comm.get('created_utc'),
            'text': comm.get('body', ''),
            'platform': 'reddit'
        })
        
    return pd.DataFrame(data)

def ingest_bluesky_data(jsonl_file):
    df = pd.read_json(jsonl_file, lines=True)
    # Extract needed columns based on Bluesky schema. 
    # Usually id, text, createdAt are present. We'll simplify for the test/pipeline.
    # We will map whatever timestamp column exists.
    timestamp_col = 'createdAt' if 'createdAt' in df.columns else 'timestamp'
    text_col = 'text' if 'text' in df.columns else 'record' # sometimes record is nested, but let's assume flat or we extract
    
    # Needs actual extraction logic based on real data structure
    extracted = []
    for _, row in df.iterrows():
        # Handle nested uri/cid or record if necessary. Assuming raw text for now or simple mapping.
        text_val = row.get('text', '')
        if isinstance(text_val, dict) and 'text' in text_val:
            text_val = text_val['text']
            
        extracted.append({
            'id': row.get('uri', str(row.get('id'))),
            'parent_id': None, # Reply to can be complex, ignore for now
            'post_type': 'post',
            'timestamp': row.get(timestamp_col),
            'text': text_val,
            'platform': 'bluesky'
        })
        
    return pd.DataFrame(extracted)

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Replace URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'http', text)
    # Replace Usernames (@username)
    text = re.sub(r'@\w+', '@user', text)
    return text

def split_and_filter_sentences(text, entity_dict):
    if not isinstance(text, str) or not text.strip():
        return []
        
    # Split text reasonably well using regex (dots, exclamation, question marks)
    # This is a naive split but handles most social media texts well enough.
    text_cleaned = re.sub(r'([.?!])\s+', r'\1<SPLIT>', text)
    sentences = text_cleaned.split('<SPLIT>')
    filtered = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        matched_category = None
        
        # Check for entity matches
        for category, entities in entity_dict.items():
            for entity in entities:
                # Use regex for word boundary to avoid partial matches
                if re.search(r'\b' + re.escape(entity.lower()) + r'\b', sentence_lower):
                    matched_category = category
                    break
            if matched_category:
                break
                
        if matched_category:
            filtered.append({
                'text': sentence.strip(),
                'entity_category': matched_category
            })
            
    return filtered
