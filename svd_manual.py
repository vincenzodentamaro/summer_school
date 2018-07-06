import numpy as np
import matplotlib.pyplot as plt
import pickle

def low_rank_approx(u=None,s=None,v=None, rank=1):
    Ar = np.zeros((len(u), len(v)))
    for i in xrange(rank):
        Ar += s[i] * np.outer(u.T[i], v[i])
    return Ar

def plot(X):
    plt.title("Beacons Components")
    for i in range(X.shape[1]):
        plt.plot(X[:i], linestyle='solid')
    plt.show()

if __name__ == "__main__":
    with open('matrix.pickle', 'rb') as handle:
        matrix = pickle.load(handle)
        plot(matrix)

    u, s, v = np.linalg.svd(matrix)
    sum = np.sum(s)
    i = 1

    while i <= np.min(matrix.shape):
        y = low_rank_approx(u,s,v, rank=i)
        print 'Signal ' + str(s[i-1] / sum *100)+'%'
        i += 1
        plot(y)
