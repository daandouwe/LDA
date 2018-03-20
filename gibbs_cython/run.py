from time import time
import sys

import numpy as np
import numba
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, dok_matrix

from lda import LDA
from toy_dataset import generate_documents
from nips import load_nips
from utils import load_20newsgroups, load_mnist, print_topics

def draw_matrix(path, matrix, transpose=False):
    fig, ax = plt.subplots(figsize=(80, 10))
    if transpose:
        matrix = matrix.T
    plt.imshow(1-matrix, cmap='Greys')
    plt.axis('off')
    plt.savefig(path)

def save_matrix(path, matrix):
    np.save(path, matrix)

def clock_time(s):
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return int(h), int(m), int(s)

def save_likelihood(likelihoods):
    with open('toy-dataset/likelihoods.txt', 'w') as f:
        for log_pred, log_joint in likelihoods:
            print(log_pred, log_joint, file=f)

ndocuments = 1000
nwords = 10000
ntopics = 20
niters = 1500

save_path = 'nips/'
save_every  = 100

# words, tf = load_20newsgroups(nwords, ndocuments)
words, tf = load_20newsgroups()
# words, tf = load_nips(ndocs=ndocuments)

ndocuments, nwords = tf.shape
print("ndocuments : {}\nnwords : {}".format(ndocuments, nwords))

lda = LDA(ntopics, alpha=0.1)

lda.run_gibbs_sampler(tf, niters)
phi = lda.phi()
theta = lda.theta()

save_matrix(save_path + 'phi.npy', phi)
save_matrix(save_path + 'theta.npy', theta)

draw_matrix(save_path + 'phi.pdf', phi)
draw_matrix(save_path + 'theta.pdf', theta, transpose=True)
print('\nFound topics:')
print_topics(words, phi, ntopics)
