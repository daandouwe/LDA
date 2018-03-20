from time import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, dok_matrix

from lda import LDA
from utils import load_20newsgroups
from toy_dataset import generate_documents

def clock_time(s):
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return int(h), int(m), int(s)

def draw_phi(phi):
    print('Drawing phi')
    ntopics, vocab_size = phi.shape
    n = int(np.sqrt(vocab_size))
    for i in range(ntopics):
        phi_i = np.reshape(phi[i], (n,n))
        plt.imshow(phi_i, cmap='Greys')
        plt.axis('off')
        plt.savefig('toy-dataset/phi-{}.pdf'.format(i))

def save_likelihood(likelihoods):
    with open('toy-dataset/likelihoods.txt', 'w') as f:
        for log_pred, log_joint in likelihoods:
            print(log_pred, log_joint, file=f)

def save_matrix(path, matrix):
    if not path.endswith('.npy'):
        path += '.npy'
    np.save(path, matrix)

n = 5
ntopics = 2*n
vocab_size = n**2
ndocs = 1000
niters = 1000
burn_in = 300
lag = 30

tf = generate_documents(ndocs, vocab_size, alpha=1)
tf = csr_matrix(tf)

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

        # if i > burn_in and i % lag == 0:
        #     topic_assignments = lda.B
        #     save_matrix('samples/B_{}'.format(i), topic_assignments)

        print("| Iteration {} | Likelihood {:.2f} | {:.2f} sec/iter | eta {}h{}m{}s".format(i, log_pred, avg_time, h, m, s))

    phi = lda.phi()
    draw_phi(phi)
    save_likelihood(likelihoods)

except KeyboardInterrupt:
    phi = lda.phi()
    draw_phi(phi)
    save_likelihood(likelihoods)
