import torch
from eval import gpuutils

def get_token_length(text, tokenizer):
    tokens = tokenizer(text, truncation=False)
    return len(tokens["input_ids"])

def get_embedding(text, model, tokenizer, device, max_len=512):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(device) # tokenize text
    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state[:, 0, :].squeeze() # using CLS embedding to represent the entire text



def get_embedding_pool(text, model, tokenizer, device='cpu', max_len = None, aggregate="max"):
    # max_len for model input 512 for bert if longer then that then pool it

    if max_len is None:
        return get_embedding(text, model, tokenizer, device, max_len)
    else:
        tokens = tokenizer(text, return_tensors="pt", truncation=False)
        token_length = tokens["input_ids"].size(1) # get length so we can see if we need to get embeddings for each portion of input (max len 512 tokens before splitting)
        print(f"Token length: {token_length}")

        chunk_embeddings = []
        if token_length > max_len:
            for i in range(0, token_length, max_len):
                chunk = text[i:i + max_len] #NOTE: maybe should have these overlapping?
                inputs = tokenizer(chunk, return_tensors="pt", max_length=max_len, truncation=True).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)

                chunk_embeddings.append(outputs.last_hidden_state[:, 0, :].squeeze())
        
            chunk_embeddings = torch.stack(chunk_embeddings)
            if aggregate == "mean":
                return chunk_embeddings.mean(dim=0).cpu()
            elif aggregate == "max":
                return chunk_embeddings.mean(dim=0).cpu()
        else:
            # its shorter then max length so send it
            return get_embedding(text, model, tokenizer, device, max_len)

    