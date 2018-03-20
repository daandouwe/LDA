from time import time

import numpy as np
from scipy.sparse import csr_matrix, dok_matrix
from scipy.special import gammaln, digamma
from scipy.misc import logsumexp

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

def normalize(x):
    return x / np.sum(x, axis=-1, keepdims=True)

class LDA(object):

    def __init__(self, ntopics, alpha=0.1, eta=0.1):
        self.ntopics = ntopics
        self.alpha = alpha
        self.eta = eta

    def initialize(self, matrix):
        self.matrix = matrix
        self.ndocs, self.nwords = matrix.shape
        self.N = np.squeeze(np.array(matrix.sum(axis=1))) # numner of words in each document d
        self.phi = {d : {n : normalize(np.random.random(self.ntopics)) for n in range(self.N[d])}
                        for d in range(self.ndocs)}
        # self.gamma = normalize(np.random.random((self.ndocs, self.ntopics)))
        # self.lmbda = np.random.random((self.ntopics, self.nwords))
        self.gamma = np.ones((self.ndocs, self.ntopics))
        self.lmbda = np.ones((self.ntopics, self.nwords))

        self.diff_gamma = 0

    def update_phi(self, d, n, w):
        new_phi = np.exp(digamma(self.gamma[d,:].T)
                            + digamma(self.lmbda[:,w])
                            - digamma(np.sum(self.lmbda, axis=1)))
        self.phi[d][n] = new_phi / np.sum(new_phi)

    def update_gamma(self, d):
        s = np.vstack([self.phi[d][n] for n in range(self.N[d])])
        self.gamma[d] = self.alpha + np.sum(s, axis=0)

    def update_lmbda(self):
        s = np.zeros((self.ntopics, self.nwords), dtype=np.float64)
        for d in range(self.ndocs):
            for n, w in enumerate(word_indices(self.matrix, d)):
                s[:,w] += self.phi[d][n]
        self.lmbda = self.eta + s

    def stochastic_update_lmbda(self, d):
        s = np.zeros((self.ntopics, self.nwords), dtype=np.float64)
        for n, w in enumerate(word_indices(self.matrix, d)):
            s[:,w] += self.ndocs * self.phi[d][n]
        return self.eta + s

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

    def vi(self, niters, threshold=0.1):
        print('Inference with batch VI.')
        elbos = []
        for step in range(niters):
            i = 0
            for d in range(self.ndocs):
                # For each document update the local variational
                # paramaters until convergence
                self.gamma[d] = np.ones(self.ntopics)
                gamma_prev = np.copy(self.gamma[d])
                converged = False
                while not converged:
                    for n, w in enumerate(word_indices(self.matrix, d)):
                        self.update_phi(d, n, w)
                        self.update_gamma(d)
                    # Check for convergence in the distribution over topics (gamma)
                    gamma_diff = np.mean(np.abs(gamma_prev - self.gamma[d]))
                    gamma_prev = np.copy(self.gamma[d])
                    i += 1
                    if gamma_diff < threshold:
                        converged = True
            # Update global variational parameters
            self.update_lmbda()

            elbo = self.elbo()
            elbos.append(elbo)
            print('Step {} | Elbo {:.2f} | reps/doc {} |'.format(step, elbo, i/self.ndocs))
        return elbos

    def stochastic_vi(self, niters, threshold=0.1, kappa=0.1, tau=1):
        print('Inference with stochastic VI.')
        elbos = []
        for step in range(niters):
            rho = (tau + step)**(-kappa)
            d = np.random.randint(0, self.ndocs)
            gamma_prev = np.copy(self.gamma[d])
            i = 0
            converged = False
            while not converged:
                for n, w in enumerate(word_indices(self.matrix, d)):
                    self.update_phi(d, n, w)
                    self.update_gamma(d)
                i += 1
                gamma_diff = np.mean(np.abs(gamma_prev - self.gamma[d]))
                gamma_prev = np.copy(self.gamma[d])
                if gamma_diff < threshold:
                    converged = True

            lmbda = self.stochastic_update_lmbda(d)
            self.lmbda = (1 - rho)*self.lmbda + rho*lmbda

            elbo = self.elbo()
            elbos.append(elbo)
            print('Step {}/{} | Elbo {:.2f} | Document {} | Rho {:.4f} | Reps {} |'.format(step, niters, elbo, d, rho, i))
        return elbos
