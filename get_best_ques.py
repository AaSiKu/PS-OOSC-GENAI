from colbert.modeling.checkpoint import Checkpoint
from colbert.infra import ColBERTConfig
from colbert.modeling.colbert import colbert_score
import pandas
import numpy
import torch

def get_top_10(ques_lis, summ_text):
    ckpt = Checkpoint("jinaai/jina-colbert-v1-en", colbert_config=ColBERTConfig(root="experiments"))
    Q = ckpt.queryFromText(ques_lis, bsize=16)
    D = ckpt.docFromText(summ_text, bsize=32)[0]
    D_mask = numpy.ones(D.shape[:2], dtype=int)
    D_mask = torch.tensor(D_mask)
    sc_lis = []
    for q in Q:
      q = q.unsqueeze(0)
      score = torch.sum(colbert_score(q, D, D_mask))
      sc_lis.append(score)
    def get_top_n_strings(list1, list2, n=10):
        paired_list = list(zip(list1, list2))
        sorted_pairs = sorted(paired_list, key=lambda x: x[0], reverse=True)
        top_n_strings = [string for _, string in sorted_pairs[:n]]
        return top_n_strings
    top_10_strings = get_top_n_strings(sc_lis, ques_lis, n=10)
    return top_10_strings