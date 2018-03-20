from time import time

import numpy as np
from scipy.sparse import csr_matrix, dok_matrix
from scipy.special import gammaln

from _lda import _run_gibbs_sampler, _initialize_chain

def log_beta(alpha, K=None):
    if K is None:
        return np.sum(gammaln(alpha), axis=1) - gammaln(np.sum(alpha, axis=1))
    else:
        return K * gammaln(alpha) - gammaln(K * alpha)

class LDA(object):

    def __init__(self, ntopics, alpha=0.1, beta=0.1):
        self.ntopics = ntopics
        self.alpha = alpha
        self.beta = beta

    def initialize_chain(self, matrix, cython=True):
        self.ndocs, self.nwords = matrix.shape
        self.N = np.squeeze(np.array(matrix.sum(axis=1), dtype=np.int32)) # numner of words in each document d
        self.M = np.zeros(self.ntopics, dtype=np.int32) # numner of words assigned to each topic k
        self.A = np.zeros((self.ndocs, self.ntopics), dtype=np.int32)
        self.B = np.zeros((self.ntopics, self.nwords), dtype=np.int32)
        self.z = np.zeros((matrix.sum(), 4), dtype=np.int32)

        # Pass to the cython code for the init
        _initialize_chain(matrix.toarray(), self.M, self.A, self.B, self.z,
                    self.ndocs, self.nwords, self.ntopics)

    def phi(self):
        phi_tilde = self.B + self.beta
        return phi_tilde / np.sum(phi_tilde, axis=1, keepdims=True)

    def theta(self):
        theta_tilde = self.A + self.alpha
        return theta_tilde / np.sum(theta_tilde, axis=1, keepdims=True)

    def log_predictive(self):
        # log p(w|z)
        log_prob = np.sum(log_beta(self.B + self.beta))
        log_prob -= self.ntopics * log_beta(self.beta, self.nwords)
        return log_prob

    def log_joint(self):
        # log p(w|z)
        log_prob = self.log_predictive()

        # log p(z)
        log_prob += np.sum(log_beta(self.A + self.alpha))
        log_prob -= self.ndocs * log_beta(self.alpha, self.ntopics)

        return log_prob

    def run_gibbs_sampler(self, matrix, niters, print_every=1):
        self.initialize_chain(matrix)
        # Pass to the Cython code
        _run_gibbs_sampler(self.N, self.M, self.A, self.B, self.z,
                self.alpha, self.beta, niters, print_every=print_every)
