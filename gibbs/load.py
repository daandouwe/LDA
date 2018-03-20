from time import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups

from read_nips import read_nips

ndocuments = 2000
nwords = 1000
ntopics = 20
niters = 1500

newsgroups = True
nips = not newsgroups

print("Loading dataset...")
t0 = time()
dataset = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'))

data_samples = dataset.data[:ndocuments]
print("done in %0.3fs." % (time() - t0))

# Use term-frequency (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=nwords,
                                stop_words='english')
t0 = time()
tf = tf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))


def print_topics(words, phi, ntopics, ntopwords=10, probs=False):
    for t in range(ntopics):
        message = "Topic #{}: ".format(t)
        if probs:
            message += " | ".join([(words[i] + ' ' + str(round(phi[t,i], 3)))
                                for i in phi[t].argsort()[:-ntopwords - 1:-1]])
        else:
            message += " ".join([words[i]
                                for i in phi[t].argsort()[:-ntopwords - 1:-1]])

        print(message)

def load_vocab(path):
    vocab = []
    with open(path, 'r') as f:
        for line in f:
            vocab.append(line.strip())
    return vocab

if newsgroups:
    phi1 = np.load('matrices/run1-phi.npy')
    phi2 = np.load('matrices/run2-phi.npy')
    phi3 = np.load('matrices/run3-phi.npy')

    words = tf_vectorizer.get_feature_names()

    print_topics(words, phi1, ntopics)
    print()
    print_topics(words, phi2, ntopics)
    print()
    print_topics(words, phi3, ntopics)

if nips:
    phi = np.load('nips/phi.npy')

    vocab = load_vocab('nips/vocab.txt')

    print_topics(vocab, phi, ntopics)
