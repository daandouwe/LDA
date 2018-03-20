from time import time

import numpy as np
from scipy.sparse import csr_matrix, dok_matrix
from scipy.special import gammaln, digamma
from scipy.misc import logsumexp

from _lda import _update_phi, _update_gamma

def word_indices(matrix, d):
    """
    For a scipy.sparse.csr_matrix.
    """
    _, word_ids = matrix.getrow(d).nonzero()
    counts = matrix[d, word_ids].toarray()[0]
    return [i for i, count in zip(word_ids, counts) for reps in range(count)]

def word_indices_(matrix, d):
    """
    For a regular np.array.
    """
    row = matrix[d,:]
    return [i for i, count in enumerate(row) for reps in range(count) if count > 0]

def log_beta(alpha, K=None):
    if K is None:
        return np.sum(gammaln(alpha), axis=1) - gammaln(np.sum(alpha, axis=1))
    else:
        return K * gammaln(alpha) - gammaln(K * alpha)

class LDA(object):

    def __init__(self, ntopics, alpha=0.1, eta=0.1):
        self.ntopics = ntopics
        self.alpha = alpha
        self.eta = eta

    def initialize(self, matrix):
        self.matrix = matrix
        self.ndocs, self.nwords = matrix.shape
        self.N = np.squeeze(np.array(matrix.sum(axis=1))) # numner of words in each document d
        self.maxwords = self.N.max()
        self.phi =  np.random.random((self.ndocs, self.maxwords, self.ntopics))
        self.gamma = np.random.random((self.ndocs, self.ntopics))
        self.lmbda = np.random.random((self.ntopics, self.nwords))

    def update_phi(self, d, n, w):
        _update_phi(self.phi, self.gamma, self.lmbda,
                    d, n, w,
                    self.ntopics, self.nwords)

    def update_gamma(self, d):
        _update_gamma(self.gamma, self.phi, self.alpha,
                    self.N[d], self.ntopics, d)

    def update_lmbda(self):
        s = np.zeros((self.ntopics, self.nwords), dtype=np.float64)
        for d in range(self.ndocs):
            for n, w in enumerate(word_indices(self.matrix, d)):
                s[:,w] += self.phi[d][n]
        self.lmbda = self.eta + s

    def elbo(self):
        elbo = 0.0
        for d in range(self.ndocs):
            elbo += (gammaln(self.ntopics * self.alpha) - self.ntopics * gammaln(self.alpha) +
                        (self.alpha - 1) * self.ntopics * (np.sum(digamma(self.gamma[d]) - digamma(np.sum(self.gamma[d])))))

            elbo += np.sum([self.phi[d][n] *  np.sum(digamma(self.gamma[d]) - digamma(np.sum(self.gamma[d])))
                                for n in range(self.N[d])])

            # elbo += missing...

            elbo -= (gammaln(np.sum(self.gamma[d])) + np.sum(gammaln(self.gamma[d])) -
                         np.sum((self.gamma[d] - 1) * (digamma(self.gamma[d]) - digamma(np.sum(self.gamma[d])))))

            elbo -= np.sum([self.phi[d][n] * np.log(self.phi[d][n]) for n in range(self.N[d])])

        return -elbo

    def vi(self, niters):
        elbos = []
        for step in range(niters):
            for d in range(self.ndocs):
                for n, w in enumerate(word_indices(self.matrix, d)):
                    self.update_phi(d,n,w)
                    self.update_gamma(d)
            self.update_lmbda()

            elbo = self.elbo()
            elbos.append(elbo)
            print('Step {} | Elbo {:.2f} |'.format(step, elbo))
        return elbos
