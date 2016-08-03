import numpy as np

def get_correlation_matrices(covariances):
    corr_matrices = np.zeros(covariances.shape)
    for i in xrange(covariances.shape[0]):
        for gene1 in xrange(covariances.shape[1]):
            for gene2 in xrange(gene1, covariances.shape[2]):
                norm_factor = (covariances[i, gene1, gene1] * covariances[i, gene2, gene2])**0.5
                corr_matrices[i, gene1, gene2] = covariances[i, gene1, gene2]/norm_factor
                corr_matrices[i, gene2, gene1] = corr_matrices[i, gene1, gene2]
    
    return corr_matrices

def get_degrees_by_gene(corr_matrices):
    degrees = np.zeros([corr_matrices.shape[0], corr_matrices.shape[1]])
    for i in xrange(corr_matrices.shape[0]):
        for gene in xrange(corr_matrices.shape[1]):
            degrees[i, gene] = abs(corr_matrices[i, gene, :]).sum()
    return degrees

