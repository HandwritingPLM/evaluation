import os
from glob import glob
import torch

#### gpuit device ####
def get_gpu_most_memory():
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0: return 'cpu' # this means there are no gpus
    free_memories = []
    for i in range(num_gpus):
        device = torch.device(f'cuda:{i}')
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        cached_memory = torch.cuda.memory_reserved(device) # this should be memory cach exchange
        
        free_memory = total_memory - (allocated_memory + cached_memory)
        
        free_memories.append(free_memory)
    best_gpu = max(enumerate(free_memories),key=lambda x: x[1])[0]
        
    return f'cuda:{best_gpu}'

#### get embeddings ####

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



#### cosine sim ####
import os
from glob import glob
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity



def get_cosine_sim_bert(gt_arr, pred_arr, model=None, tokenizer=None, device=None):
    if model is None: model = BertModel.from_pretrained("bert-base-uncased") # using a default model
    if tokenizer is None: tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    if device is None: device = get_gpu_most_memory() # gets gpu with most free mem or returns cpu if no gpus
  
    model.to(device)
    model.eval()

    #### get embeddings ####
    gt_embeddings = []
    pred_embeddings = []
  
    for text in gt_arr:
        embedding = get_embedding_pool(text, model=model, tokenizer=tokenizer, device=device)
        gt_embeddings.append(embedding.cpu().detach())  

    for text in pred_arr:
        embedding = get_embedding_pool(text, model=model, tokenizer=tokenizer, device=device)
        pred_embeddings.append(embedding.cpu().detach())  

    #### compute cosine similarity ####
    gt_embeddings = torch.stack(gt_embeddings).numpy()
    pred_embeddings = torch.stack(pred_embeddings).numpy()
    similarities = cosine_similarity(gt_embeddings, pred_embeddings)
    diagonal_similarities = similarities.diagonal() 

    return diagonal_similarities.tolist()  





def get_cosine_sim_bert_single(gt, pred, model, tokenizer, device):
    gt_embedding = get_embedding_pool(gt, model=model, tokenizer=tokenizer, device=device)
    pred_embedding = get_embedding_pool(pred, model=model, tokenizer=tokenizer, device=device)
    similarity = cosine_similarity(gt_embedding.cpu().reshape(1, -1), pred_embedding.cpu().reshape(1, -1))
    return(similarity[0])