from sklearn.cluster import KMeans
from pyroconductor import corpcor, glasso
import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy import stats
from scipy import optimize
from scipy import sparse
from sklearn.neighbors import kneighbors_graph



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

class PrincipalTree(object):
    def __init__(self, num_nodes, sigma=1, lam=1, 
                 tol=0.001, max_iter=100):
        self.max_iter =  max_iter
        self.num_nodes = num_nodes
        self.sigma = sigma
        self.tol = tol
        self.lam = lam
        self.principal_point_generator = KMeansPPGenerator(self.num_nodes)
    
    def fit(self, data_array):
        undersampler = KMeansPPGenerator(data_array.shape[0]/2)
        undsmpl_data_array = undersampler.generate_points(data_array)
        n_samples = undsmpl_data_array.shape[0]
        F = self.principal_point_generator.generate_points(undsmpl_data_array).T
        self.initial_points = np.copy(F.T)
        R = np.zeros([n_samples, self.num_nodes])
        ones_vec_num_nodes = np.ones([self.num_nodes])
        ones_vec_n_samples = np.ones([n_samples])
        for curr_iter in xrange(self.max_iter):
            D = squareform(pdist(F.T))
            G = nx.Graph(D)
            T = nx.minimum_spanning_tree(G)
            B = nx.adjacency_matrix(T, weight=None).todense()
            L = np.diag(np.diag(np.dot(B, ones_vec_num_nodes))) - B
            for i in xrange(n_samples):
                Z = np.array([np.exp(-np.linalg.norm(undsmpl_data_array[i,:] - F[:, j])/self.sigma)
                             for j in xrange(self.num_nodes)])
                R[i, :] = Z/Z.sum()
            
            Lam = np.diag(np.dot(R.T, ones_vec_n_samples))
            Inv = np.linalg.inv(self.lam*L + Lam)
            next_F = np.dot(undsmpl_data_array.T, np.dot(R, Inv))
            if curr_iter % 10 == 0:
                print 'In iteration %s' % curr_iter
                print np.linalg.norm(F - next_F)
            if np.linalg.norm(F - next_F) < self.tol:
                break
            F = next_F
        self.node_positions = np.array(next_F.T)
        self.tree = T
        


class PrincipalGraph(object):
    def __init__(self, n_nodes, max_iter=10, eps=1e-5, gstruct='l1-graph', lam=1., gamma=0.5, sigma=0.01,
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
            node_positions = init_node_pos;
        
        n_samples, n_dims = data_array.shape
        
        n_nodes = node_positions.shape[0]
        
        if G is not None:
            G = sklearn.neighbors.kneighbors_graph(node_positions, self.nn).todense()

        
        # construct only once for all iterations
        if self.gstruct == 'l1-graph':
    
            # low triangular sum_i sum_{j < i}
            [row, col] = np.tril_indices_from(G)
            nw = len(row)
            nvar = nw + n_nodes * n_dims;

            rc = {}
            for i in range(len(row)):
                key_ij = row[i] + col[i] * n_nodes
                rc[key_ij] =  i    
            
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
                    a[:, pos_ij] = np.vstack([-data_array[j, :], data_array[j ,:]])
                start_i = nw + (i - 1) * n_dims + 1
                end_i = start_i + n_dims - 1
                a[:, start_i:end_i] = np.vstack([-np.eye[n_dims, n_dims], -eye[n_dims, n_dims]])
                bb = np.vstack([-data_array[i, :], data_array[i, :]])
                if i == 0:
                    A = np.copy(a)
                    b = np.copy(bb)
                else:
                    A = sparse.vstack([A, a]);
                    b = np.vstack([b, bb]);

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

        
class BranchingMorphingMixture(object):

    def __init__(self, num_nodes=None, cov_estimator='corpcor', cov_reg=None, tol=0.001, max_iter=10,
                 step_size=0.01, sigma=0.01, lam=1., gamma=0.5):
        self.num_nodes = num_nodes
        self.cov_reg = cov_reg
        self.cov_estimator = cov_estimator
        self.tol = tol
        self.max_iter = max_iter
        self.step_size = step_size
        self.sigma = sigma
        self.lam = lam
        self.gamma = gamma

    def fit(self, data_array):
        if self.num_nodes is None:
            self.num_nodes = data_array.shape[0]/2
        pt = PrincipalGraph(gstruct='span-tree', gamma=self.gamma, sigma=self.sigma, 
                            lam=self.lam, n_nodes=self.num_nodes)
        pt.fit(data_array)
        init_means = pt.node_positions
        self._pt = pt
        self.tree = pt.graph
        self.node_distances = self._calculate_node_distances(self.tree)
        init_covs = np.zeros([self.num_nodes, data_array.shape[1], data_array.shape[1]])
        for i in xrange(self.num_nodes):
            init_covs[i, :, :] = np.eye(data_array.shape[1])

        self._coord_asc_loop(data_array, init_means, init_covs)

    def _calculate_node_distances(self, tree):
        shortest_paths = nx.all_pairs_shortest_path_length(self.tree)
        node_distances = np.zeros([self.num_nodes, self.num_nodes])
        for i in xrange(self.num_nodes):
            for j in xrange(self.num_nodes):
                node_distances[i, j] = shortest_paths[i][j]
        del(shortest_paths)
        node_distances /= node_distances.max()
        return node_distances

    def smooth(self, means, covariances):
        #TODO Need to think how to smooth properly: probably by taking N nearest neighbors on
        # tree and doing some spline fitting
        return means, covariances

    def _map_samples_to_nodes(self, data_array, means, covs):
        mapping = np.zeros([data_array.shape[0]]) - 1
        mapping_probs = np.zeros([data_array.shape[0], means.shape[0]])
        for i in xrange(means.shape[0]):
            mapping_probs[:, i] = stats.multivariate_normal.pdf(data_array, mean=means[i, :], 
                                                                   cov=covs[i, :, :], allow_singular=True)
        
        for i in xrange(data_array.shape[0]):
            mapping[i] = np.argmax(mapping_probs[i, :])
        return mapping, mapping_probs

    
    def _calculate_gaussian_params_from_mapping(self, data_array, mapping, mapping_probs):
        means = np.zeros([self.num_nodes, data_array.shape[1]])
        covariances = np.zeros([self.num_nodes, data_array.shape[1], data_array.shape[1]])
        n_samples = mapping_probs.shape[0]
        normalized_probs = mapping_probs / mapping_probs.sum(axis=1)[:, None]
        for i in xrange(self.num_nodes):
            expected_distances = np.array([np.dot(normalized_probs[j, :], self.node_distances[i, :])
                                           for j in range(n_samples)])
            #distances = np.array([self.node_distances[i, j] for j in mapping])
            
            #weights = np.exp(-0.5*(expected_distances/(self.step_size))**2)
            weights = 1./((1 + expected_distances)**2 * self.step_size)
            weights = np.reshape(weights, [len(weights), 1])
            print weights
            weights /= weights.sum()
            weighted_data = weights * data_array
            means[i, :] = weighted_data.sum(axis=0)
            if self.cov_reg is None:
                covariances[i, :, :] = np.copy(corpcor.cov_shrink(data_array, weights=weights))
            else:
                covariances[i, :, :] = np.copy(corpcor.cov_shrink(data_array, weights=weights, 
                                                                  **{'lambda':self.cov_reg}))
        return self.smooth(means, covariances)

        
    def _coord_asc_loop(self, data_array, init_means, init_covs):
        curr_mapping, curr_mapping_probs = self._map_samples_to_nodes(data_array,
                                                                      init_means, init_covs)
        for i in xrange(self.max_iter):
            print 'ITER %s' % i
            curr_means, curr_covs = self._calculate_gaussian_params_from_mapping(data_array, 
                                                                                 curr_mapping, 
                                                                                 curr_mapping_probs)
            prev_mapping_probs = curr_mapping_probs
            curr_mapping, curr_mapping_probs = self._map_samples_to_nodes(data_array, 
                                                                          curr_means, curr_covs)
            if ((prev_mapping_probs - curr_mapping_probs)**2).sum() < self.tol:
                break
                print 'Done!'
        self.mean = curr_means
        self.covariances = curr_covs
        self.mapping_probs = curr_mapping_probs


    def predict_proba(self, data_array):
        mapping, mapping_probs = self._map_samples_to_nodes(data_array, self.means, self.covs)
        return mapping, mapping_probs

    def predict(self, data_array):
        mapping, _ = self.predict_proba(data_array, self.means, self.covs)
        return mapping
