from collections import defaultdict, Counter
import numpy as np
import networkx as nx
import random

from sklearn.covariance import OAS
from sklearn.mixture import GMM

from . import log
from . import morphing_mixture
from . import model_comparisons
from . import utils

class MetastableGraph(object):
    def __init__(self, state_model, state_edges, edge_weights={}):
        self.state_edges = state_edges
        self.edge_weights = edge_weights
        self.state_model = state_model
    
    def get_mfest(self):
        G = nx.Graph()
        log_probs = self.state_model.score(self.state_model.centroids)
        for n1, n2 in self.state_edges:
            weight = abs(log_probs[n1.index] - log_probs[n2.index])
            #weight = norm(self.state_model.centroids[n1] - self.state_model.centroids[n2])
            G.add_edge(n1, n2, weight=weight)
        T = nx.minimum_spanning_tree(G, weight='weight')
        return T.edges()
    
    def map_data_to_states(self, data_array):
        memberships = self.state_model.predict(data_array)
        data_to_state = {}
        state_to_data = defaultdict(list)
        for i, idx in enumerate(memberships):
            state_idx = utils.create_state_index(idx)
            data_to_state[i] = state_idx          
            state_to_data[state_idx].append(i) 
        return data_to_state, state_to_data

    def fit_transition_model(self, data_array, states=None, 
                            fit_type='spline', degree=3, 
                            n_components=5, n_samples=10, 
                            method='principal_curve'):
        if states is None:
            data_to_analyze = data_array
        else:
            data_to_state, _ = self.map_data_to_states(data_array)
            analyzed_indices = [i for i in xrange(data_array.shape[0]) 
                                if data_to_state[i].index in states or data_to_state[i].color in states]
            data_to_analyze = np.array(data_array[analyzed_indices, :])
        trace, model_opt, log_p_opt = morphing_mixture.sample_morphing_gaussian_mixtures(data_to_analyze, 
                fit_type=fit_type, degree=degree, 
                n_components=n_components, n_iters=n_samples, 
                method=method)
        if states is None:
            return model_opt, trace
        else:
            return model_opt, trace, analyzed_indices

def _gmm_from_memberships(data, memberships, covariance_type):
    clusters = set(memberships)
    n_clusters = len(clusters)
    gmm = GMM(n_components=n_clusters, params='m')
    gmm.weights_ = np.ones([n_clusters])/n_clusters
    gmm.means_ = np.zeros([n_clusters, data.shape[1]]) 
    if covariance_type == 'diag':
        gmm.covars_ = np.zeros([n_clusters, data.shape[1]])
    if covariance_type == 'spherical':
        gmm.covars_ = np.zeros([n_clusters])
    if covariance_type == 'full':
        gmm.covars_ = np.zeros([n_clusters, data.shape[1], data.shape[1]])

    for cluster in clusters:
        cluster = int(cluster)
        indices = (memberships == cluster)
        gmm.means_[cluster, :] = data[indices, :].mean(axis=0)
        if covariance_type in ['diag', 'spherical']:
            #TODO Fix covariance calculation, for now, return cov=1
            #D = np.diag(np.cov(data[indices, :].T))
            D = np.ones([data.shape[1]])
            if covariance_type == 'spherical':
                gmm.covars_[cluster] = D.mean()
            else:
                gmm.covars_[cluster] = D
        if covariance_type == 'full':
            cov_estimator = OAS()
            cov_estimator.fit(data[indices, :])
            gmm.covars_[cluster] = cov_estimator.covariance_
    return gmm


def get_gmm_metastable_graph(data, n_states=None, covariance_type='diag',
                             n_init=20, 
                             connection_estimation_method='max_path_distance_diff', 
                             min_paths=3, memberships=None):

    if covariance_type not in ['diag', 'spherical', 'full']:
        raise ValueError('Invalid covariance type %s' % covariance_type)
    if memberships is None:
        if n_states is None:
            MAX_STATES = 10
            MIN_SAMPLES = 3
            bics = np.zeros([MAX_STATES])
            bics[:] = np.inf
            for curr_states in range(2, MAX_STATES):
                gmm = GMM(n_components=curr_states, params='mc', 
                        covariance_type=covariance_type, n_init=n_init)
                gmm.fit(data)
                found_singleton = False
                for state, sample_count in Counter(gmm.predict(data)).iteritems():
                    if sample_count < MIN_SAMPLES:
                        found_singleton = True
                        break
                if not found_singleton:
                    bics[curr_states] = gmm.bic(data)
            n_states = int(np.argmin(bics))

        gmm = GMM(n_components=n_states, params='mc', 
            covariance_type=covariance_type, n_init=n_init)
        gmm.fit(data)
    else:
        gmm = _gmm_from_memberships(data, memberships, covariance_type)
    gmm.centroids = gmm.means_
    state_edges = utils.get_metastable_connections_from_gmm(data, gmm, 
                                                            connection_estimation_method=connection_estimation_method,
                                                            min_paths=min_paths)
    return MetastableGraph(gmm, [(utils.create_state_index(i), utils.create_state_index(j)) for i, j in state_edges])

def get_gmm_bootstrapped_metastable_graph(data_array, n_states=None, n_boot=100, 
                                       memberships=None, **kwargs):
    metastable_graphs = []
    data_sample_size = data_array.shape[0]
    full_metastable_graph = get_gmm_metastable_graph(data_array, n_states, 
                                                           memberships=memberships, **kwargs)

    n_states = full_metastable_graph.state_model.centroids.shape[0]
    for boot_iter in xrange(n_boot):
        sample_indices = [random.choice(range(data_sample_size)) for _ in xrange(data_sample_size)]
        if memberships is None:
            sample_memberships = None
        else:
            sample_memberships = memberships[sample_indices]
        metastable_graphs.append(get_gmm_metastable_graph(data_array[sample_indices, :], n_states, 
                                                                         memberships=sample_memberships, **kwargs))
    bootstrap_edge_fractions = model_comparisons.get_edge_fractions([full_metastable_graph] + metastable_graphs)
    return full_metastable_graph, metastable_graphs, bootstrap_edge_fractions
    

