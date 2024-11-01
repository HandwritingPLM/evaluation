import torch
from eval import gpuutils

def get_embedding(text, model, tokenizer, device=gpuutilsget_gpu_most_memory):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512) # tokenize text
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs.last_hidden_state[:, 0, :].squeeze() # using CLS embedding to represent the entire text