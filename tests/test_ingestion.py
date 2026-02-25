import pandas as pd
from src.ingestion import ingest_reddit_data, ingest_bluesky_data, preprocess_text, split_and_filter_sentences


def test_ingest_reddit_data(tmp_path):
    import json
    # Dummy Submissions
    subs = [
        {"id": "s1", "created_utc": 1700000000, "title": "Fire breaks out in LA", "selftext": "FEMA is responding."},
        {"id": "s2", "created_utc": 1700001000, "title": "No text", "selftext": ""}
    ]
    sub_file = tmp_path / "subs.json"
    with open(sub_file, "w") as f:
        json.dump(subs, f)

    # Dummy Comments
    comms = [
        {"id": "c1", "parent_id": "t3_s1", "created_utc": 1700000500, "body": "They are doing a great job!"},
        {"id": "c2", "parent_id": "t1_c1", "created_utc": 1700000600, "body": "LAFD says otherwise."}
    ]
    comm_file = tmp_path / "comms.json"
    with open(comm_file, "w") as f:
        json.dump(comms, f)

    df = ingest_reddit_data(sub_file, comm_file)

    assert len(df) == 4
    assert list(df.columns) == ["id", "parent_id", "post_type", "timestamp", "text", "platform"]
    assert df.iloc[0]["post_type"] == "submission"
    assert df.iloc[0]["text"] == "Fire breaks out in LA FEMA is responding."
    assert df.iloc[2]["post_type"] == "comment"
    assert df.iloc[2]["parent_id"] == "t3_s1"

def test_preprocess_text():
    text = "Hey @user123 check out this link: https://t.co/xyz"
    cleaned = preprocess_text(text)
    assert cleaned == "Hey @user check out this link: http"

def test_split_and_filter_sentences():
    text = "The fire is bad. LAFD is here. FEMA gave us water. What about the power company?"
    entities = {
        "emergency_management": ["FEMA", "LA County", "emergency management", "evacuation order", "sheriff"],
        "first_responders": ["firefighter", "paramedic", "first responder", "CAL FIRE", "LAFD"],
        "utilities": ["PG&E", "Southern California Edison", "SCE", "Edison", "power company", "utility", "PSPS"]
    }
    
    sentences = split_and_filter_sentences(text, entities)
    assert len(sentences) == 3
    assert sentences[0]["text"] == "LAFD is here."
    assert sentences[0]["entity_category"] == "first_responders"
    
    assert sentences[1]["text"] == "FEMA gave us water."
    assert sentences[1]["entity_category"] == "emergency_management"

    assert sentences[2]["text"] == "What about the power company?"
    assert sentences[2]["entity_category"] == "utilities"

if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdirname:
        test_ingest_reddit_data(Path(tmpdirname))
    test_preprocess_text()
    test_split_and_filter_sentences()
    print("All tests passed!")
