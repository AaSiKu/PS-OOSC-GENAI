from colbert.modeling.checkpoint import Checkpoint
from colbert.infra import ColBERTConfig
import pandas
import numpy
import torch
import json

def summ_scraped_lis(scraped_data):
    for i in range(len(scraped_data)):
      dic=[]
      for keywords in keys:
        summary = summarize_text(scraped_data[i], keywords, top_n=1)
        dic.append((keywords, summary))
      dic2 = [i[1] for i in dick]
      dic3 = [' '.join(j) for j in dic2]
      scraped_data[i]=" ".join(dic3)
    return scraped_data

def get_top_n_indices(arr, n=5):
    arr = np.asarray(arr)
    sorted_indices = np.argsort(arr)
    top_n_indices = sorted_indices[-n:]
    
    return list(top_n_indices)


def top_5_main():
    ckpt = Checkpoint("jinaai/jina-colbert-v1-en", colbert_config=ColBERTConfig(root="experiments"))
    file_path = './Url_Title_Content.json'
    with open(file_path, 'r') as file:
        god_raccoon = json.load(file)
    scraped_data = [i["content"] for i in god_raccoon]
    scraped_data = summ_scraped_lis(scraped_data)
    D = ckpt.docFromText(scraped_data, bsize=32)[0]
    D_mask = numpy.ones(D.shape[:2], dtype=int)
    D_mask = torch.tensor(D_mask)

    Q = ckpt.queryFromText([top_10_strings[0]], bsize=16)
    score = numpy.array(colbert_score(Q, D, D_mask))
    for i in top_10_strings[1:]:
        Q = ckpt.queryFromText([i], bsize=16)
        score += numpy.array(colbert_score(Q, D, D_mask))
    top_n_ind = get_top_n_indices(score)

    cute_raccoon = []

    for i in top_n_ind:
        temp_raccoon = dict()
        temp_raccoon["url"] = god_raccoon[i]["url"]
        temp_raccoon["title"] = god_raccoon[i]["title"]
        cute_raccoon.append(temp_raccoon)

    return cute_raccoon