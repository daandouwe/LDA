import numpy as np
import matplotlib.pyplot as plt

def generate_topics(n):
    topics = np.zeros((2*n, n, n), dtype=np.float64)
    for i in range(n):
        topics[i, i, :] = 1 / n
        topics[n + i, :, i] = 1 / n
    return topics

def plot_topics(topics):
    k, n, n = topics.shape
    for i in range(k):
        plt.imshow(topics[i,:,:], cmap='Greys')
        plt.axis('off')
        plt.savefig("toy-dataset/topic-{}.pdf".format(i))

def plot_documents(docs, k=10):
    ndocs, vocab_size = docs.shape
    w = int(np.sqrt(vocab_size))
    for i in range(k):
        doc = np.reshape(docs[i,:], (w,w))
        plt.imshow(doc, cmap='Greys')
        plt.axis('off')
        plt.savefig("toy-dataset/doc-{}.pdf".format(i))

def draw_dirichlet(ndocs, vocab_size, alpha):
    alphas = alpha * np.ones(vocab_size)
    return np.random.dirichlet(alphas, ndocs)

def draw_multinomial(p_z):
    return np.random.multinomial(1, p_z).argmax()

def generate_documents(ndocs, vocab_size, nwords = 100, alpha=0.1):
    topics = generate_topics(int(np.sqrt(vocab_size)))
    topics = np.reshape(topics, (-1, vocab_size))
    docs = np.zeros((ndocs, vocab_size), dtype=np.int32)
    ntopics = topics.shape[0]
    p_z = draw_dirichlet(ndocs, ntopics, alpha=alpha)
    for d in range(ndocs):
        doc = np.zeros(vocab_size)
        for i in range(nwords):
            k = draw_multinomial(p_z[d])
            p_w = np.squeeze(np.reshape(topics[k], (1, -1)))
            w = draw_multinomial(p_w)
            doc[w] += 1
        docs[d] = doc
    return docs

if __name__ == '__main__':

    n = 5
    ntopics = 2*n
    vocab_size = n**2
    ndocs = 2000

    topics = generate_topics(n)
    plot_topics(topics)

    docs = generate_documents(ndocs, vocab_size, alpha=1)
    plot_documents(docs)
