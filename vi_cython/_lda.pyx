#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import cython
# from cython cimport double

import numpy as np
cimport numpy as np
from numpy.math cimport EULER

from libc.math cimport exp, log
from libc.stdlib cimport rand, RAND_MAX
from libc.stdio cimport printf, fflush, stdout, stderr, fprintf
from libc.time cimport clock, clock_t, CLOCKS_PER_SEC

# cimport scipy.special.cython_special.psi as digamma

def _update_phi(double[:,:,:] phi, double[:,:] gamma, double[:,:] lmbda,
                int d, int n, int w,
                int ntopics, int nwords):

    cdef int k, v
    cdef double total = 0
    cdef double sum_lmbda = 0
    cdef double[:] phi_ = np.zeros(ntopics)

    for k in range(ntopics):
        # print("phi {}/{}".format(k, ntopics))
        sum_lmbda = 0.0
        for v in range(nwords):
            sum_lmbda += lmbda[k, v]
        phi_[k] = exp(digamma(gamma[d, k])
                    + digamma(lmbda[k, w])
                    - digamma(sum_lmbda))
        # phi_[k] = exp(gamma[d, k]
                    # + lmbda[k, w]
                    # - sum_lmbda)

    for k in range(ntopics):
        phi[d, n, k] = phi_[k] / total


def _update_gamma(double[:,:] gamma, double[:,:,:] phi, double alpha,
            int N, int ntopics, int d):
    cdef int k, n
    cdef double new_gamma
    for k in range(ntopics):
        # print("gamma {}/{}".format(k, ntopics))

        new_gamma = alpha
        for n in range(N):
            new_gamma += phi[d, n, k]
        gamma[d, k] = new_gamma


cdef double digamma(double x):
    # print("digamma!")
    if x <= 1e-6:
        # psi(x) = -EULER - 1/x + O(x)
        return -EULER - 1. / x

    cdef double r, result = 0

    # psi(x + 1) = psi(x) + 1/x
    while x < 6:
        result -= 1. / x
        x += 1

    # psi(x) = log(x) - 1/(2x) - 1/(12x**2) + 1/(120x**4) - 1/(252x**6)
    #          + O(1/x**8)
    r = 1. / x
    result += log(x) - .5 * r
    r = r * r
    result -= r * ((1./12.) - r * ((1./120.) - r * (1./252.)))
    return result

# cdef double digamma(double x) nogil:
#     """
#     Source: https://gist.github.com/miksu/223d81add9df8f878d75d39caa42873f
#
#     Copyright (c) 1995-2004 by Radford M. Neal
#     Permission is granted for anyone to copy, use, modify, or distribute this
#     program and accompanying programs and documents for any purpose, provided
#     this copyright notice is retained and prominently displayed, along with
#     a note saying that the original programs are available from Radford Neal's
#     web page, and note is made of any changes made to the programs.  The
#     programs and documents are distributed without any warranty, express or
#     implied.  As the programs were written for research purposes only, they have
#     not been tested to the degree that would be advisable in any important
#     application.  All use of these programs is entirely at the user's own risk.
#     ----------------------------------------------------------------------------
#     digamma(x) is defined as (d/dx) log Gamma(x).  It is computed here
#     using an asymptotic expansion when x>5.  For x<=5, the recurrence
#     relation digamma(x) = digamma(x+1) - 1/x is used repeatedly.  See
#     Venables & Ripley, Modern Applied Statistics with S-Plus, pp. 151-152.
#     ----------------------------------------------------------------------------
#     Note:
#     This is a direct Cython translation of the original digamma function.
#     Original programs are available from Radfort Neal's webs page:
#       http://www.cs.toronto.edu/~radford/fbm.software.html
#     Original C implementation is also available at:
#       https://github.com/mrquincle/fbm/blob/master/util/digamma.c
#     """
#
#     cdef double r, f, t
#
#     r = 0
#
#     while (x<=5):
#         r -= 1/x
#         x += 1
#
#     f = 1/(x*x)
#
#     t = f*(-1/12.0 + f*(1/120.0 + f*(-1/252.0 + f*(1/240.0 + f*(-1/132.0
#         + f*(691/32760.0 + f*(-1/12.0 + f*3617/8160.0)))))))
#
#     return r + log(x) - 0.5/x + t
