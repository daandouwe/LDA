from time import time
import sys

import numpy as np
import numba
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, dok_matrix

from lda import LDA
from utils import load_20newsgroups
from toy_dataset import generate_documents
from read_nips import read_nips

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

ndocuments = 2000
nwords = 1000
ntopics = 20
niters = 1500

save_path = 'nips/'
save_every  = 100

# tf = load_20newsgroups(nwords, ndocuments)
vocab, tf = read_nips()
ndocuments, nwords = tf.shape

lda = LDA(ntopics, alpha=0.1)

gibbs_iterator = lda.gibbs_sampler(tf, niters)

likelihoods = []
try:
    total_time = 0
    for i in range(niters):
        t0 = time()
        # One full Gibbs iteration
        next(gibbs_iterator)

        # Timing
        elapsed = time() - t0
        total_time += elapsed
        avg_time = total_time / (i+1)
        h, m, s = clock_time(niters * avg_time - total_time)

        # Likelihood
        log_pred = lda.log_predictive()
        log_joint = lda.log_joint()
        likelihoods.append((log_pred, log_joint))

        if i % save_every == 0:
            phi = lda.phi()
            theta = lda.theta()
            save_matrix(save_path + 'phi.npy', phi)
            save_matrix(save_path + 'theta.npy', theta)

        print("| Iteration {} | Likelihood {:.2f} | {:.2f} sec/iter | eta {}h{}m{}s".format(i, log_pred, avg_time, h, m, s))

    draw_matrix(save_path + 'phi.pdf', phi)
    draw_matrix(save_path + 'theta.pdf', theta, transpose=True)
    save_likelihood(likelihoods)

except KeyboardInterrupt:
    save_matrix(save_path + 'phi.npy', phi)
    save_matrix(save_path + 'theta.npy', theta)

    draw_matrix(save_path + 'phi.pdf', phi)
    draw_matrix(save_path + 'theta.pdf', theta, transpose=True)
    save_likelihood(likelihoods)
