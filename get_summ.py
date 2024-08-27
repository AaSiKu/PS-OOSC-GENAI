## Imports
import numpy as np
import pandas as pd
import re
import networkx as nx

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

# Basic Preprocessing
def preprocess(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove stopwords
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

def extract_keywords(article_text):
    processed_docs = [preprocess(i) for i in article_text.split('.')]
    vectorizer = TfidfVectorizer(max_features=1000)
    processed_docs = [i for i in processed_docs if len(i)>1]
    X = vectorizer.fit_transform(processed_docs)

    nmf_model = NMF(n_components=30, random_state=42)
    W = nmf_model.fit_transform(X)
    H = nmf_model.components_

    feature_names = vectorizer.get_feature_names_out()

    def display_topics(H, feature_names, num_top_words):
        lis = []
        for topic_idx, topic in enumerate(H):
            #print(f"Topic {topic_idx}:")
            #print(", ".join([feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]))
            lis.append([feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]])
        return lis

    keys = display_topics(H, feature_names, num_top_words=10)
    return keys

# Preprocessing text: tokenize and clean sentences
def preprocess_text(text):
    # Sentence tokenization
    sentences = sent_tokenize(text)
    return sentences

def text_rank(text, keys):
    # Clean and tokenize a sentence
    def preprocess_sentence(sentence):
        sentence = sentence.lower()
        sentence = re.sub(r'\d+', '', sentence)
        sentence = re.sub(r'[^\w\s]', '', sentence)
        return sentence

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

    # Extract summary
    def summarize_text(text, keywords, top_n=3):
        sentences = preprocess_text(text)
        similarity_matrix = build_similarity_matrix(sentences, keywords)
        graph = build_graph(similarity_matrix)
        sentence_ranks = textrank(graph)
        ranked_sentences = sorted(((sentence_ranks[i], s) for i, s in enumerate(sentences)), reverse=True)
        summary_lis = [sentence.strip('\t').strip('\n').strip('\n\n') for _, sentence in ranked_sentences[:top_n]]
        return summary_lis

    dic = []

    for keywords in keys:
      summary = summarize_text(text, keywords, top_n=5)
      dic.append((keywords, summary))

    return dic