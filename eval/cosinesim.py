import os
from glob import glob
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

from eval import getembeddings
from eval import gpuutils

import importlib
importlib.reload(getembeddings)


# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = BertModel.from_pretrained("bert-base-uncased")

# device = gpuutils.get_gpu_most_memory()
# model.to(device)


# gt = "I dont want no food"
# pred = "No I dont want food"

def get_cosine_sim_bert(gt_arr, pred_arr, model=None, tokenizer=None, device=None):
    if model is None: model = BertModel.from_pretrained("bert-base-uncased") # using a default model
    if tokenizer is None: tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    if device is None: device = gpuutils.get_gpu_most_memory() # gets gpu with most free mem or returns cpu if no gpus
  
    model.to(device)
    model.eval()

    #### get embeddings ####
    gt_embeddings = []
    pred_embeddings = []
  
    for text in gt_arr:
        embedding = getembeddings.get_embedding_pool(text, model=model, tokenizer=tokenizer, device=device)
        gt_embeddings.append(embedding.cpu().detach())  

    for text in pred_arr:
        embedding = getembeddings.get_embedding_pool(text, model=model, tokenizer=tokenizer, device=device)
        pred_embeddings.append(embedding.cpu().detach())  

    #### compute cosine similarity ####
    gt_embeddings = torch.stack(gt_embeddings).numpy()
    pred_embeddings = torch.stack(pred_embeddings).numpy()
    similarities = cosine_similarity(gt_embeddings, pred_embeddings)
    diagonal_similarities = similarities.diagonal() 

    return diagonal_similarities.tolist()  





def get_cosine_sim_bert_single(gt, pred, model, tokenizer, device):
    gt_embedding = getembeddings.get_embedding_pool(gt, model=model, tokenizer=tokenizer, device=device)
    pred_embedding = getembeddings.get_embedding_pool(pred, model=model, tokenizer=tokenizer, device=device)
    similarity = cosine_similarity(gt_embedding.cpu().reshape(1, -1), pred_embedding.cpu().reshape(1, -1))
    return(similarity[0])