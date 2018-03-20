#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import cython

import numpy as np
cimport numpy as np

from cython.parallel import prange
from libc.stdlib cimport rand, RAND_MAX
from libc.stdio cimport printf, fflush, stdout, stderr, fprintf
from libc.time cimport clock, clock_t, CLOCKS_PER_SEC


cdef int searchsorted(double[:] arr, int length, double value) nogil:
    """Bisection search (c.f. numpy.searchsorted)
    Find the index into sorted array `arr` of length `length` such that, if
    `value` were inserted before the index, the order of `arr` would be
    preserved.
    """
    cdef int imin, imax, imid
    imin = 0
    imax = length
    while imin < imax:
        imid = imin + ((imax - imin) >> 2)
        if value > arr[imid]:
            imin = imid + 1
        else:
            imax = imid
    return imin

def _initialize_chain(int[:,:] matrix, int[:] M, int[:,:] A, int[:,:] B, int[:,:] z,
            int ndocs, int nwords, int ntopics):

    cdef int pair, count, rep
    cdef int d, i, w, k
    pair = 0
    for d in range(ndocs):
        i = 0 # i-th word in document d
        for w in range(nwords):
            count = matrix[d,w]
            if count == 0:
                pass
            else:
                for rep in range(count):
                    k = <int>((ntopics-1) * (<double> rand() / RAND_MAX))
                    M[k] += 1
                    A[d,k] += 1
                    B[k,w] += 1
                    z[pair,0], z[pair,1], z[pair,2], z[pair,3] = d, i, w, k
                    i += 1
                    pair += 1


def _run_gibbs_sampler(int[:] N, int[:] M, int[:,:] A, int[:,:] B, int[:,:] z,
            double alpha, double beta, int niters, int print_every):
    """
    During sampling:
        d is the document number
        i is the i-th word in document d
        w is the word index of the i-th word in document d
        k is the previous topic assignment of the i-th word in document d
    """
    cdef int step, pair, d, i, w, k
    cdef int ntopics = B.shape[0]
    cdef int nwords = B.shape[1]
    cdef int npairs = z.shape[0]
    cdef double theta, phi, cumsum, r
    cdef double[:] p_z = np.zeros(ntopics)

    cdef clock_t t0 = clock()
    cdef double elapsed, eta
    cdef int h0, m0, s0, h1, m1, s1

    for step in range(niters):

        for pair in range(npairs):
            d, i, w, k = z[pair,0], z[pair,1], z[pair,2], z[pair,3]

            M[k] -= 1
            A[d,k] -= 1
            B[k,w] -= 1

            cumsum = 0.0
            for i in range(ntopics):
                theta = (A[d,i] + alpha) / (N[d] - 1.0 + ntopics * alpha)
                phi = (B[i,w] + beta) / (M[i] + nwords * beta)
                cumsum += theta * phi
                p_z[i] = cumsum

            # Draw new topic assignment
            r = cumsum * (<double> rand() / RAND_MAX) # random number in [0,..,cumsum-1]
            k = searchsorted(p_z, ntopics, r) # find index for r in the ordered array p_z

            M[k] += 1
            A[d,k] += 1
            B[k,w] += 1

            z[pair,0], z[pair,1], z[pair,2], z[pair,3] = d, i, w, k

        if (step+1) % print_every == 0:
            # Calculate elapsed time and estimated time of arrival and print to screen
            elapsed = <double> (clock() - t0) / CLOCKS_PER_SEC
            h0 = <int> (elapsed / 3600)
            m0 = <int> ((elapsed / 60) % 60)
            s0 = <int> (elapsed % 60)
            eta = niters * elapsed / (step+1) - elapsed
            h1 = <int> (eta / 3600)
            m1 = <int> ((eta / 60) % 60)
            s1 = <int> (eta % 60)
            fprintf(stderr, "\r| Step %d/%d. | elapsed %dh:%dm:%ds | eta %dh:%dm:%ds | ",
                                      step+1, niters, h0, m0, s0, h1, m1, s1);
