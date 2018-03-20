# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Lars Buitinck
#         Chyi-Kwei Yau <chyikwei.yau@gmail.com>
# License: BSD 3 clause

from __future__ import print_function
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups

import numpy as np

n_documents = 2000
n_words = 1000
n_topics = 10
n_top_words = 20


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


# Load the 20 newsgroups dataset and vectorize it. We use a few heuristics
# to filter out useless terms early on: the posts are stripped of headers,
# footers and quoted replies, and common English words, words occurring in
# only one document or in at least 95% of the documents are removed.

print("Loading dataset...")
t0 = time()
dataset = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'))

data_samples = dataset.data[:n_documents]
print("done in %0.3fs." % (time() - t0))


# Use term-frequency (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_words,
                                stop_words='english')
t0 = time()
tf = tf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))
print(type(tf))
print(tf.nnz)
print(tf.shape)
print(100 * tf.nnz / np.prod(tf.shape))

Nd = np.sum(tf, axis=1)
print(Nd)

# print("Fitting LDA models with tf features, "
#       "n_documents=%d and n_words=%d..."
#       % (n_documents, n_words))
# lda = LatentDirichletAllocation(n_components=n_topics, max_iter=5,
#                                 learning_method='online',
#                                 learning_offset=50.,
#                                 random_state=0)
# t0 = time()
# lda.fit(tf)
# print("done in %0.3fs." % (time() - t0))
#
# print("\nTopics in LDA model:")
# tf_feature_names = tf_vectorizer.get_feature_names()
# print(tf_get_feature_names)
# print_top_words(lda, tf_feature_names, n_top_words)

def print_documents(n):
    words = tf_vectorizer.get_feature_names()
    for d in range(n):
        print("\nDocument", d)
        document = np.squeeze(tf[d,:].toarray())
        for i, count in enumerate(document):
            if count > 0:
                print(words[i], count)
