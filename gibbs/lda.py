from time import time

import numpy as np
from scipy.sparse import csr_matrix, dok_matrix
from scipy.special import gammaln

def sample_index(p):
    """
    Sample from the Multinomial distribution and return the sample index.
    """
    # return np.random.multinomial(1,p).argmax()
    return np.searchsorted(np.cumsum(p), np.random.rand())

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

    def __init__(self, ntopics, alpha=0.1, beta=0.1):
        self.ntopics = ntopics
        self.alpha = alpha
        self.beta = beta

    def initialize_chain(self, matrix):
        print('Initializing markov chain.')
        self.ndocs, self.nwords = matrix.shape
        self.N = np.squeeze(np.array(matrix.sum(axis=1))) # numner of words in each document d
        self.M = np.zeros(self.ntopics, dtype=np.int64) # numner of words assigned to each topic k
        self.A = np.zeros((self.ndocs, self.ntopics), dtype=np.int64)
        self.B = np.zeros((self.ntopics, self.nwords), dtype=np.int64)
        self.z = dict() # topic of word i in document d, z[(d,i)] = in [0,1,...,ntopics-1]
        self.z_ = dict() # topic of word i in document d, z[(d,i)] = in [0,1,...,ntopics-1]

        for d in range(self.ndocs):
            for i, w in enumerate(word_indices(matrix, d)):
                k = np.random.randint(self.ntopics)
                self.M[k] += 1
                self.A[d,k] += 1
                self.B[k,w] += 1
                self.z[d,i] = k
                self.z_[(d,i,w)] = k

    def phi(self):
        phi_tilde = self.B + self.beta
        return phi_tilde / np.sum(phi_tilde, axis=1, keepdims=True)

    def theta(self):
        theta_tilde = self.A + self.alpha
        return theta_tilde / np.sum(theta_tilde, axis=1, keepdims=True)

    def conditional_dist(self, d, w):
        theta = (self.A[d,:] + self.alpha) / (self.N[d] - 1 + self.ntopics * self.alpha)
        phi = (self.B[:,w].T + self.beta) / (self.M + self.nwords * self.beta)
        p_tilde = theta * phi
        return np.squeeze(p_tilde / np.sum(p_tilde))

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

    def gibbs_step(self, d, i, w, k):
        """
        A single marginal gibbs sample.
            d: the document number
            i: the i-th word in document d
            w: the word index of the i-th word in document d
            k: the previous previous topic assignment of the i-th word in document d
        """
        self.M[k] -= 1
        self.A[d,k] -= 1
        self.B[k,w] -= 1

        p_z = self.conditional_dist(d, w)
        k = sample_index(p_z)

        self.M[k] += 1
        self.A[d,k] += 1
        self.B[k,w] += 1

        return k

    def gibbs_sampler_(self, matrix, niters):
        """
        NOTE: slower!
        Iterator over gibbs-sampling steps.
            d is the document number
            i is the i-th word in document d
            w is the word index of the i-th word in document d
            k is the previous topic assignment of the i-th word in document d
        """
        self.initialize_chain(matrix)
        for step in range(niters):
            for d in range(self.ndocs):
                for i, w in enumerate(word_indices(matrix, d)):
                    k = self.z[d,i]
                    k = self.gibbs_step(d, i, w, k)
                    self.z[d,i] = k
            yield

    def gibbs_sampler(self, matrix, niters):
        """
        Returns an iterator.
            d is the document number
            i is the i-th word in document d
            w is the word index of the i-th word in document d
            k is the previous topic assignment of the i-th word in document d
        """
        self.initialize_chain(matrix)
        for step in range(niters):
            for (d, i, w), k in self.z_.items():
                k = self.gibbs_step(d, i, w, k)
                self.z_[d,i,w] = k
            yield
