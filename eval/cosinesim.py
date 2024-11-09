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

def get_cosine_sim_bert(gt, pred, model, tokenizer, device):
    gt_embedding = getembeddings.get_embedding_pool(gt, model=model, tokenizer=tokenizer, device=device)
    pred_embedding = getembeddings.get_embedding_pool(pred, model=model, tokenizer=tokenizer, device=device)
    similarity = cosine_similarity(gt_embedding.cpu().reshape(1, -1), pred_embedding.cpu().reshape(1, -1))
    return(similarity[0])