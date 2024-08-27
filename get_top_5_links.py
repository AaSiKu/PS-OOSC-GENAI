from colbert.modeling.checkpoint import Checkpoint
from colbert.infra import ColBERTConfig
import pandas
import numpy as np
import torch
import networkx as nx
import json
from get_summ import extract_keywords, preprocess_text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

from colbert.modeling.checkpoint import Checkpoint
from colbert.infra import ColBERTConfig
from colbert.modeling.colbert import colbert_score

# Build a similarity matrix
def build_similarity_matrix(sentences, keywords):
    # Use TF-IDF to compute word importance
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(sentences)
    # Compute cosine similarity between each pair of sentences
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    # Boost similarity for sentences containing keywords
    keyword_boost = np.zeros(similarity_matrix.shape)
    for i, sentence in enumerate(sentences):
        for keyword in keywords:
            if keyword in sentence:
                keyword_boost[i] += 1
    # Integrate the keyword boost into the similarity matrix
    similarity_matrix += keyword_boost
    return similarity_matrix
# Build a graph from the similarity matrix
def build_graph(similarity_matrix):
    graph = nx.from_numpy_array(similarity_matrix)
    return graph
# Rank sentences using TextRank
def textrank(graph, damping=0.85, max_iter=100):
    return nx.pagerank(graph, alpha=damping, max_iter=max_iter)

def summarize_text(text, keywords, top_n=3):
    sentences = preprocess_text(text)
    similarity_matrix = build_similarity_matrix(sentences, keywords)
    graph = build_graph(similarity_matrix)
    sentence_ranks = textrank(graph)
    ranked_sentences = sorted(((sentence_ranks[i], s) for i, s in enumerate(sentences)), reverse=True)
    summary_lis = [sentence.strip('\t').strip('\n').strip('\n\n') for _, sentence in ranked_sentences[:top_n]]
    return summary_lis

def summ_scraped_lis(scraped_data):
    for i in range(len(scraped_data)):
      dic=[]
      keys = extract_keywords(scraped_data[i])
      for keywords in keys:
        summary = summarize_text(scraped_data[i], keywords, top_n=1)
        dic.append((keywords, summary))
      dic2 = [i[1] for i in dic]
      dic3 = [' '.join(j) for j in dic2]
      scraped_data[i]=" ".join(dic3)
    return scraped_data

def get_top_n_indices(arr, n=5):
    arr = np.asarray(arr)
    sorted_indices = np.argsort(arr)
    top_n_indices = sorted_indices[-n:]
    
    return list(top_n_indices)


def top_5_main(top_10_strings):
    ckpt = Checkpoint("jinaai/jina-colbert-v1-en", colbert_config=ColBERTConfig(root="experiments"))
    file_path = './Url_Title_Content.json'
    with open(file_path, 'r') as file:
        god_raccoon = json.load(file)
    scraped_data = [i["content"] for i in god_raccoon]
    scraped_data = summ_scraped_lis(scraped_data)
    D = ckpt.docFromText(scraped_data, bsize=32)[0]
    D_mask = np.ones(D.shape[:2], dtype=int)
    D_mask = torch.tensor(D_mask)

    Q = ckpt.queryFromText([top_10_strings[0]], bsize=16)
    score = np.array(colbert_score(Q, D, D_mask))
    for i in top_10_strings[1:]:
        Q = ckpt.queryFromText([i], bsize=16)
        score += np.array(colbert_score(Q, D, D_mask))
    top_n_ind = get_top_n_indices(score)

    cute_raccoon = []

    for i in top_n_ind:
        temp_raccoon = dict()
        temp_raccoon["url"] = god_raccoon[i]["url"]
        temp_raccoon["title"] = god_raccoon[i]["title"]
        cute_raccoon.append(temp_raccoon)

    return cute_raccoon