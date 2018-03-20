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
ndocs = 100
niters = 30

tf = generate_documents(ndocs, vocab_size, alpha=0.1)
tf = csr_matrix(tf)

lda = LDA(ntopics, alpha=0.1)

lda.initialize(tf)

# elbos = lda.vi(niters)
elbos = lda.stochastic_vi(100)

phi = lda.lmbda / np.sum(lda.lmbda, keepdims=True)
draw_phi(phi)

with open('elbo.txt', 'w') as f:
    for elbo in elbos:
        print(elbo, file=f)
