from time import time

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups, fetch_mldata

def load_20newsgroups(nwords=None, ndocuments=-1, max_df=0.95, min_df=2):
    print("Loading 20 newsgroups dataset...")
    t0 = time()
    dataset = fetch_20newsgroups(shuffle=True, random_state=1,
                                 remove=('headers', 'footers', 'quotes'))

    data_samples = dataset.data[:ndocuments]
    print("done in %0.3fs." % (time() - t0))

    # Use term-frequency (raw term count) features for LDA.
    print("Extracting tf features for LDA...")
    tf_vectorizer = CountVectorizer(max_df=max_df, min_df=min_df,
                                    max_features=nwords,
                                    stop_words='english')
    t0 = time()
    tf = tf_vectorizer.fit_transform(data_samples)
    print("done in %0.3fs." % (time() - t0))
    words = tf_vectorizer.get_feature_names()

    return words, tf

def load_mnist(ndocuments=-1):
    print("Loading MNIST dataset...")
    mnist = fetch_mldata('MNIST original')
    idx = np.random.randint(mnist.data.shape[0], size=ndocuments) # for random selection of rows
    return mnist.data[idx].astype(np.int32)


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
