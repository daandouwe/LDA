from time import time

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

def load_nips(ndocs=-1):
    print('Reading in NIPS dataset.')
    nips = pd.read_csv('NIPS_1987-2015.csv').as_matrix()
    vocab, tf = nips[:,0], nips[:,1:].astype(np.int32).T
    del nips
    if ndocs > 0:
        tf = tf[-ndocs:,:]
    return vocab, csr_matrix(tf)


if __name__ == '__main__':
    vocab, tf = load_nips()
    with open('nips/vocab.txt', 'w') as f:
        for word in vocab:
            print(word, file=f)
