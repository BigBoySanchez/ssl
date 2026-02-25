# Step-by-Step Plan: Emotion Detection Pipeline

**Objective:** Implement an emotion detection pipeline for social media text targeting specific entities over time, focusing on disaster management discourse (specifically the 2025 LA wildfires).

## Phase 1: Data Ingestion and Preprocessing
1. **Load Data by Platform**:
   * *Reddit*: Read JSON files, explicitly separating original submissions and comments. Ensure timestamps and texts are extracted.
   * *Bluesky*: Read JSONL/CSV files extracting the text and timestamp.
2. **Text Cleaning**:
   * Replace all usernames with the generic placeholder `@user`.
   * Replace all hyperlinks/URLs with the generic placeholder `http`.
   * *(This cleaning step mirrors preprocessing from models like Twitter RoBERTa that handle social media text).*
3. **Define Entity Lexicons**: Create dictionary lists to capture all entity references.
   * *Emergency Management*: "FEMA", "LA County", "emergency management", "evacuation order", "sheriff".
   * *First Responders*: "firefighter(s)", "paramedic(s)", "first responder(s)", "CAL FIRE", "LAFD".
   * *Utilities*: "PG&E", "Southern California Edison", "SCE", "Edison", "power company", "utility", "PSPS".
4. **Sentence Splitting & Filtering**: 
   * Segment each social media post into individual sentences. This is crucial to prevent mixed emotions (e.g., "PG&E failed. Firefighters are heroes") from muddying the analysis.
   * Filter the dataset to retain *only* the sentences that contain at least one string match from the target entity lexicons. 

## Phase 2: Model Inference
1. **Initialize Hugging Face Pipeline**:
   * Import `pipeline` from `transformers` (`text-classification`).
   * Load Model: `Hidden-States/roberta-base-goemotions` (trained on Reddit, natively outputs 28 classes).
   * Load Tokenizer: `Hidden-States/roberta-base-goemotions`.
   * Set pipeline arguments: `top_k=None`, `truncation=True`.
2. **Run Inference**: 
   * Pass the filtered, entity-containing sentences through the pipeline.
   * This generates a probability distribution across 28 GoEmotions categories (27 emotions + neutral).
3. **Single-Label Extraction**: 
   * Convert the multi-label probability distributions into a single dominant emotion per sentence by applying `argmax(probabilities)`.
   * You will now have one clean emotion label per sentence, directly tied to the entity mentioned in that sentence.

## Phase 3: Aggregation and Matrix Construction
1. **Data Structuring & Emotion Grouping**:
   * Map each processed sentence to its predicted emotion, the entity class, the platform, and the timestamp.
   * Group the 28 fine-grained emotions into macro-buckets for interpretability:
     * *Negative*: Anger, annoyance, disapproval, disgust.
     * *Positive*: Gratitude, admiration, approval.
     * *Distress*: Fear, nervousness, sadness.
     * *Neutral*: Neutral (can be tracked separately or excluded depending on analysis needs).
2. **Event-Time Framing**: 
   * Discretize the timestamps into specific event phases (e.g., Day 0-2: outbreak / evacuation surge; Day 3-7: peak disruption / response; Day 8+: containment / recovery).
3. **Construct the Emotion × Entity Matrix**: 
   * Aggregate the data to build a matrix where rows are Entities and columns are macro-bucket Emotions.
   * Compute normalized percentages within each entity (e.g., "% of posts about Utilities expressing Anger" vs. "% of posts about First Responders expressing Gratitude").
   * Track these percentages across the discretized event daily time bins.
4. **Validation (Crucial Step for Research)**:
   * Draw a random stratified sample of ~300 sentences across platforms and entity groups.
   * Have 2 human annotators manually label: 1) Is the emotion directed at the target? 2) What is the single emotion label?
   * Report annotator agreement and rough model accuracy.
