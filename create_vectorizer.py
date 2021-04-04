import os
import re
import json
import spacy
import pickle
import time
import numpy as np
import en_core_web_sm

from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer


def main():
    t0 = time.time()
    print('loading spacy model')
    nlp = en_core_web_sm.load()
    corpora = []
    data_path = 'data'

    print('loading data')
    N = len(os.listdir(data_path))
    for filename in tqdm(os.listdir(data_path)):
        data = json.load(open(os.path.join(data_path, filename)))
        text = data['text']
        text = re.sub('\n', '', text)
        text_sentences = nlp(text)
        sentences = [sentence.text for sentence in text_sentences.sents]
        corpora.extend(sentences)

    print('fitting vectorizer')
    tfidf = TfidfVectorizer(sublinear_tf=True,
                           strip_accents='unicode',
                           min_df=5,
                           stop_words='english',
                           ngram_range=(1,1))
    tfidf.fit(corpora)

    print('saving vectorizer')
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)

    print('done')
    t1 = time.time()
    total = t1-t0
    print(f'running this script took {np.round(total, 2)}s')


if __name__ == "__main__":
    main()

