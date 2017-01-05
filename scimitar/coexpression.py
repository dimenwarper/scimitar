import numpy as np
from sklearn.cluster import SpectralClustering

def get_coregulatory_similarity(corr_matrices):
    n_matrices = corr_matrices.shape[0]
    similarity_matrix = np.zeros([n_matrices, n_matrices])
    for i in xrange(n_matrices):
        for j in xrange(i + 1, n_matrices):
            mat1 = abs(corr_matrices[i, :, :])
            mat2 = abs(corr_matrices[j, :, :])
            similarity_matrix[i, j] = np.linalg.norm(mat1 - mat2)
            similarity_matrix[j, i] = similarity_matrix[i, j]

    similarity_matrix /= similarity_matrix.max()
    similarity_matrix = 1 - similarity_matrix
    return similarity_matrix

def get_coregulatory_states(corr_matrices, similarity_matrix, n_clusters):
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
    labels = spectral.fit_predict(similarity_matrix)

    coreg_states = {}
    for ci in np.unique(labels):
        coreg_states[ci] = corr_matrices[labels == ci, :, :].mean(axis=0)
    return coreg_states, labels


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

