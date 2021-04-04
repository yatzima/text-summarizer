import re
import spacy
import pickle
import time
import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st
import matplotlib.pyplot as plt
import en_core_web_sm

from sklearn.metrics.pairwise import cosine_similarity

#@st.cache
def load_vectorizer(filename='vectorizer.pkl'):
    with open(filename, 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer


st.title("Text Summarizer")
st.header("Automatic summarization is the process of shortening a set of data \
computationally, to create a subset (a summary) that represents the most \
important or relevant information within the original content.")

st.sidebar.title("Settings")

language  = st.sidebar.selectbox(
    "Choose language",
    ("English", "Swedish")
)

model_choice = st.sidebar.radio("Choose model", 
        ["LexRank (Extraction-based)",
        "TextRank (Extraction-based)", 
        "Deep Learning (Abstraction-based)"])

factor = st.sidebar.slider('Factor to summarize text.', 0, 100, 75)

text  = st.text_area("Input your text here:", height=400) 

# Load stuff
vectorizer = load_vectorizer()
nlp = en_core_web_sm.load()

if text:
    text_sentences = nlp(text)
    text = re.sub('\n', '', text)
    sentences = [sentence.text for sentence in text_sentences.sents]
    N = int(np.round((factor/100)*len(sentences)))
    mapping = {i:sentence for i, sentence in enumerate(sentences)}
    M = vectorizer.transform(sentences)
    S = cosine_similarity(M)
    np.fill_diagonal(S, 0)
    adjacency_df = pd.DataFrame(S)
    G = nx.convert_matrix.from_pandas_adjacency(adjacency_df)
    G = nx.relabel_nodes(G, mapping)
    scores = nx.pagerank(G)
    score_df = pd.DataFrame(scores.items(), columns=['Descriptions', 'Score'])
    indices = score_df.sort_values(by='Score', ascending=False)[:N].index.sort_values()
    pd.set_option("display.max_colwidth", None)
    summarized = score_df.iloc[indices]['Descriptions'].to_string(header=False, index=False)
    summarized = re.sub('\n', '', summarized)
    st.text("Summary of your text:")
    st.write(summarized.strip())
    st.balloons()


