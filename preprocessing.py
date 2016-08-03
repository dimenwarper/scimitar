import numpy as np
from scipy import stats

def calculate_ordering_confidence_weights(samples):
    ngenes = samples.shape[1]
    cell_order_by_gene = np.zeros(samples.shape)
    for i in xrange(cell_order_by_gene.shape[1]):
        cell_order_by_gene[:, i] = [j for j, _ in sorted(enumerate(samples[:, i]), key=lambda x:x[1])]
    corr_M = np.zeros([ngenes, ngenes])
    for i in xrange(ngenes):
        for j in xrange(i + 1, ngenes):
            corr_M[i, j] = abs(stats.pearsonr(cell_order_by_gene[:, i], cell_order_by_gene[:, j])[0])
            corr_M[j, i] = corr_M[i, j]
    return corr_M.max(axis=0)


