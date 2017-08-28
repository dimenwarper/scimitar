from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range

import numpy as np

from scipy import stats
from scipy import optimize
from scipy import sparse

import networkx as nx

from sklearn.cluster import KMeans
from sklearn.manifold import SpectralEmbedding
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances

from pyroconductor import corpcor

def adaptive_rbf_matrix(data_array, n_neighbors=30):
    n_samples = data_array.shape[0]
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(data_array)
    A = pairwise_distances(data_array, metric='l2')
    
    n_distances = np.reshape(nn.kneighbors(data_array)[1][:, -1], (n_samples, 1))
    S = np.dot(n_distances, n_distances.T) / A.mean()
    A = np.exp(-(A + 1)/(S + 1))
    return A


class PrincipalPointGenerator(object):
    def generate_points(self, data_array):
        raise NotImplementedError('generate_points is not implemented in this class')
        
class KMeansPPGenerator(PrincipalPointGenerator):
    def __init__(self, num_points):
        self.num_points = num_points
        self.kmeans = KMeans(n_clusters=self.num_points)
    
    def generate_points(self, data_array):
        self.kmeans.fit(data_array)
        return self.kmeans.cluster_centers_


class PrincipalGraph(object):
    def __init__(self, 
                 n_nodes, 
                 max_iter=10, 
                 eps=1e-5, 
                 gstruct='l1-graph', 
                 lam=1., 
                 gamma=0.5, 
                 sigma=0.01,
                 nn=5):
        self.max_iter = max_iter
        self.eps = eps
        if gstruct not in ['l1-graph', 'span-tree']:
            raise ValueError('Graph structure %s is not supported' % gstruct)
        self.gstruct = gstruct
        self.lam = lam
        self.gamma = gamma
        self.sigma = sigma
        self.nn = nn
        self.n_nodes = n_nodes
        self.principal_point_generator = KMeansPPGenerator(self.n_nodes)
        
    
    def fit(self, data_array, G=None, init_node_pos=None):
        if init_node_pos is None:
            node_positions = self.principal_point_generator.generate_points(data_array)
        else:
            node_positions = init_node_pos
        
        n_samples, n_dims = data_array.shape
        
        n_nodes = node_positions.shape[0]
        
        if G is not None:
            G = kneighbors_graph(node_positions, self.nn).todense()

        
        # construct only once for all iterations
        if self.gstruct == 'l1-graph':
    
            # low triangular sum_i sum_{j < i}
            [row, col] = np.tril_indices_from(G)
            nw = len(row)
            nvar = nw + n_nodes * n_dims

            rc = {}
            for i in range(len(row)):
                key_ij = row[i] + col[i] * n_nodes
                rc[key_ij] = i    
            
            # construct A and b
            for i in range(n_samples):
                nn_i = np.where(G[:, i] == 1)[0]
                a = sparse.csr_matrix([2 * n_dims, nvar])
                for jj in range(len(nn_i)):
                    j = nn_i[jj]
                    key_ij = i + j * n_nodes
                    if i < j:
                        key_ij = j + i * n_nodes
                    pos_ij = rc[key_ij]
                    a[:, pos_ij] = np.vstack([-data_array[j, :], data_array[j, :]])
                start_i = nw + (i - 1) * n_dims + 1
                end_i = start_i + n_dims - 1
                a[:, start_i:end_i] = np.vstack([-np.eye[n_dims, n_dims], -np.eye[n_dims, n_dims]])
                bb = np.vstack([-data_array[i, :], data_array[i, :]])
                if i == 0:
                    A = np.copy(a)
                    b = np.copy(bb)
                else:
                    A = sparse.vstack([A, a])
                    b = np.vstack([b, bb])

        objs = []
        lp_vars = [] 
        for itr in range(self.max_iter):
            norm_sq = np.tile((node_positions.T ** 2).sum(axis=0), [n_nodes, 1])
            Phi = norm_sq + norm_sq.T - 2 * np.dot(node_positions, node_positions.T)
            if self.gstruct == 'l1-graph':
                val = np.zeros([nw, 1])
                for i in range(nw):
                    val[i] = Phi[row[i], col[i]]
                c = np.vstack([2 * val, self.lam * np.ones([n_dims * n_nodes, 1])])
            
                res = optimize.linprog(c, A, b)
            
                w_eta, obj_W = res[0], res[1]
            
                w = w_eta[:nw]
                W_tril = np.array([n_nodes, n_nodes])
                for i in range(nw):
                    W_tril[row[i], col[i]] = w[i]
                W = W_tril + W_tril.T
            
                # warm start
                lp_vars = w_eta
            
            if self.gstruct == 'span-tree':
                nxG = nx.Graph(Phi)
                t_ = nx.minimum_spanning_tree(nxG)
                stree = np.asarray(nx.to_numpy_matrix(t_))
                W = np.zeros(stree.shape)
                W[stree != 0] = 1.
                obj_W = stree.sum()
            
            P, obj_P = self._soft_assignment(data_array, node_positions)
            obj = obj_W + self.gamma * obj_P
            if itr == 0:
                objs= np.copy(obj)
            else:
                objs = np.vstack([objs, obj])
            
            print('iter=%d, obj=%f\n' % (itr, obj))
    
            if itr > 1:
                relative_diff = abs( objs[itr - 1] - obj) / abs(objs[itr - 1])
                if relative_diff < self.eps:
                    break
                if itr >= self.max_iter:
                    print('Max iteration reached')
            

            node_positions = self._generate_centers(data_array, W, P)
        self._probabilities = P
        self.node_positions = node_positions
        self.graph = nx.Graph(W)


    def _generate_centers(self, data_array, W, P):
        # prevent singular
        Q = 2 * (np.diag(W.sum(axis=0)) - W ) + self.gamma * np.diag(P.sum(axis=0))
        B = np.dot(self.gamma * data_array.T, P)
        return np.asarray(np.dot(B, np.linalg.inv(Q)).T)
    
    def _soft_assignment(self, data_array, node_positions):
        n_samples, n_dims = data_array.shape
        n_nodes = node_positions.shape[0]
        norm_X_sq = np.tile((data_array.T ** 2).sum(axis=0)[:, np.newaxis], [1, n_nodes])
        norm_C_sq = np.tile((node_positions.T ** 2).sum(axis=0), [n_samples, 1])
        
        
        dist_XC = norm_X_sq + norm_C_sq - 2 * np.dot(data_array, node_positions.T)
        
        min_dist = dist_XC.min(axis=1)
        dist_XC = dist_XC - np.tile(min_dist[:, np.newaxis], [1, n_nodes])
        Phi_XC = np.exp(-dist_XC / self.sigma);
        Phi_XC_sum = np.tile(Phi_XC.sum(axis=1)[:, np.newaxis], [1, n_nodes])
        P = Phi_XC / Phi_XC_sum

        obj = -self.sigma * sum(np.log((np.exp(-dist_XC / self.sigma)).sum(axis=1)) - (min_dist / self.sigma))
        return P, obj


class BranchedEmbeddedGaussians(object):

    def __init__(self, n_nodes=None, 
                 npcs=0.8, 
                 embedding_dims=2,
                 cov_estimator='corpcor', 
                 cov_reg=None, 
                 cov_indices=None,
                 max_iter=10,
                 sigma=0.01, 
                 lam=1., 
                 gamma=1.,
                 n_neighbors=30,
                 just_tree=False):
        self.n_nodes = n_nodes
        self.cov_reg = cov_reg
        self.cov_estimator = cov_estimator
        self.cov_indices = cov_indices
        self.max_iter = max_iter
        self.sigma = sigma
        self.lam = lam
        self.gamma = gamma
        self.npcs = npcs
        self.n_neighbors = n_neighbors
        self.embedding = SpectralEmbedding(n_components=embedding_dims, 
                                           affinity='precomputed')
        self.pca = PCA(n_components=self.npcs)
        self.just_tree = just_tree

    def fit(self, data_array):
        n_samples, n_dims = data_array.shape
        if self.n_nodes is None:
            self.n_nodes = 0.1
        if type(self.n_nodes) == float:
            self.n_nodes = max(2, np.round(n_samples * self.n_nodes))

        self._pca_tx = self.pca.fit_transform(data_array)
        self._affinity = adaptive_rbf_matrix(data_array,
                                             n_neighbors=self.n_neighbors)
        self._embedding_tx = self.embedding.fit_transform(self._affinity)

        pt = PrincipalGraph(gstruct='span-tree', gamma=self.gamma, 
                            sigma=self.sigma, max_iter=self.max_iter,
                            lam=self.lam, n_nodes=self.n_nodes)
        pt.fit(self._embedding_tx)
        self._pt = pt
        self.graph = pt.graph
        self.node_positions = self._pt.node_positions

        if self.cov_indices is None:
            cov_indices = np.arange(0, n_samples)
        else:
            cov_indices = self.cov_indices
        if not self.just_tree:
            self.means, self.covariances = self._calculate_gaussian_params(data_array, 
                                                                        self._pt._probabilities,
                                                                     cov_indices)


    def _map_samples_to_nodes(self, 
                              data_array, 
                              means, 
                              covs):
        mapping = np.zeros([data_array.shape[0]]) - 1
        mapping_probs = np.zeros([data_array.shape[0], means.shape[0]])
        for i in xrange(means.shape[0]):
            mapping_probs[:, i] = stats.multivariate_normal.pdf(data_array[:, self.cov_indices], 
                                                                mean=means[i, self.cov_indices], 
                                                                cov=covs[i, :, :], allow_singular=True)
        
        for i in xrange(data_array.shape[0]):
            mapping[i] = np.argmax(mapping_probs[i, :])
        return mapping, mapping_probs

    def _calculate_gaussian_params(self, 
                                   data_array, 
                                   mapping_probs, 
                                   cov_indices):
        means = np.zeros([self.n_nodes, data_array.shape[1]])
        cov_dim = len(cov_indices)
        covariances = np.zeros([self.n_nodes, cov_dim, cov_dim])
        for i in xrange(self.n_nodes):
            weights = mapping_probs[:, i]
            weights = np.reshape(weights, [len(weights), 1])
            weighted_data = weights * data_array
            means[i, :] = weighted_data.sum(axis=0)
            if self.cov_reg is None:
                covariances[i, :, :] = np.copy(corpcor.cov_shrink(data_array[:, cov_indices], 
                                                                  weights=weights))
            else:
                covariances[i, :, :] = np.copy(corpcor.cov_shrink(data_array[: cov_indices],
                                                                  weights=weights, 
                                                                  **{'lambda':self.cov_reg}))
        return means, covariances


    def predict_proba(self, data_array):
        mapping, mapping_probs = self._map_samples_to_nodes(data_array, 
                                                            self.means, 
                                                            self.covs)
        return mapping, mapping_probs

    def predict(self, data_array):
        mapping, _ = self.predict_proba(data_array, 
                                        self.means, 
                                        self.covs)
        return mapping
