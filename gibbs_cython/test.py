from time import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, dok_matrix

from lda import LDA
from utils import load_20newsgroups, load_mnist
from toy_dataset import generate_documents

def clock_time(s):
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return int(h), int(m), int(s)

def draw_phi(path, phi):
    print('Drawing phi.')
    ntopics, vocab_size = phi.shape
    n = int(np.sqrt(vocab_size))
    for i in range(ntopics):
        phi_i = np.reshape(phi[i], (n,n))
        plt.imshow(phi_i, cmap='Greys')
        plt.axis('off')
        plt.savefig('{}/phi-{}.png'.format(path, i))

def draw_mnist(path, m, k=10):
    print('Drawing {} samples from MNIST.'.format(k))
    ntopics, vocab_size = m.shape
    n = int(np.sqrt(vocab_size))
    for i in range(k):
        m_i = np.reshape(m[i], (n,n))
        plt.imshow(m_i, cmap='Greys')
        plt.axis('off')
        plt.savefig('{}/m-{}.png'.format(path, i))

def save_phi(path, phi):
    print('Saving phi.')
    ntopics, vocab_size = phi.shape
    n = int(np.sqrt(vocab_size))
    for i in range(ntopics):
        phi_i = np.reshape(phi[i], (n,n))
        with open('{}/phi-{}.npy'.format(path, i), 'wb') as f:
            np.save(f, phi_i)

def save_mnist(path, mnist, k=10):
    print('Drawing mnist.')
    ntopics, vocab_size = mnist.shape
    n = int(np.sqrt(vocab_size))
    for i in range(k):
        mnist_i = np.reshape(mnist[i], (n,n))
        with open('{}/m-{}.npy'.format(path, i), 'wb') as f:
            np.save(f, mnist_i)

def save_likelihood(likelihoods):
    with open('toy-dataset/likelihoods.txt', 'w') as f:
        for log_pred, log_joint in likelihoods:
            print(log_pred, log_joint, file=f)

def save_proportions(path, theta, k=10):
    with open('{}/theta.txt'.format(path), 'w') as f:
        for i in range(k):
            args = theta[i].argsort()[::-1]
            vals = np.sort(theta[i])[::-1]
            for arg, val in zip(args, vals):
                print(arg, val, file=f)
            print('\n', file=f)


def save_matrix(path, matrix):
    if not path.endswith('.npy'):
        path += '.npy'
    np.save(path, matrix)

lines = True
if lines:
    n = 5
    ntopics = 2*n
    vocab_size = n**2
    ndocs = 100
    niters = 1000
    burn_in = 300
    lag = 30
    path = 'toy-dataset'
    tf = generate_documents(ndocs, vocab_size, alpha=1)
    tf = csr_matrix(tf)
else:
    ntopics = 25
    ndocs = 1000
    niters = 1000
    path = 'mnist'

    mnist = load_mnist(ndocs)
    tf = csr_matrix(mnist)
    vocab_size = tf.shape

    draw_mnist(path, mnist, k=10)
    save_mnist(path, mnist, k=10)

lda = LDA(ntopics, alpha=0.1)

try:
    t0 = time()
    lda.run_gibbs_sampler(tf, niters, print_every=1)
    t1 = time()
    print('Finished in {:.1f} seconds ({:.4f} seconds per iteration).'.format(t1 - t0, (t1 - t0)/niters))
    draw_phi(path, lda.phi())
    save_phi(path, lda.phi())
    save_proportions('mnist', lda.theta(), k=10)

except KeyboardInterrupt:
    draw_phi(path, lda.phi())
    save_phi(path, lda.phi())
    save_proportions('mnist', lda.theta(), k=10)
