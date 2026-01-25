# Transformer Implementation Pitfalls

## CUDA Device-Side Assertions (Embedding/vocab Mismatch)

### Symptom
`CUDA error: device-side assert triggered` accompanied by `indexSelectLargeIndex` or similar index-out-of-bounds errors in the stack trace, often pointing to the first layer or embedding lookup.

### Cause
The tokenizer's vocabulary size is larger than the model's pre-trained embedding matrix size. This can happen when:
1. Using models like `vinai/bertweet-base` where the tokenizer includes extra tokens not in the base model's config.
2. Adding special tokens to the tokenizer without resizing the model embeddings.
3. Using a tokenizer from a different model version or architecture inadvertently.

### Solution
Always resize the model's token embeddings to match the tokenizer's length after initialization:

```python
# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# CRITICAL FIX
model.resize_token_embeddings(len(tokenizer))
```

This ensures that any token ID produced by the tokenizer resolves to a valid row in the embedding matrix.
