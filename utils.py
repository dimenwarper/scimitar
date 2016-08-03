from collections import defaultdict, namedtuple
import numpy as np
from numpy.linalg import norm
import networkx as nx

from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

from . import settings
from . import stats

StateIndex = namedtuple('StateIndex', 'index color')
def create_state_index(idx):
    return StateIndex(index=idx, color=settings.STATE_COLORS[idx])

def all_pairs_shortest_path_weighted(graph):
    paths = nx.all_pairs_dijkstra_path(graph, weight='weight')
    path_weighted_lengths = defaultdict(dict)
    for n1 in paths:
        for n2 in paths[n1]:
            path = paths[n1][n2]
            path_len = sum([graph[path[k-1]][path[k]]['weight'] for k in xrange(1, len(path))])
            path_weighted_lengths[n1][n2] = path_len
            path_weighted_lengths[n2][n1] = path_len
    return paths, path_weighted_lengths

def get_metastable_connections_from_gmm(data, gmm, 
                                        connection_estimation_method='max_path_distance_diff', 
                                        min_paths=3, distance='euclidean', 
                                        low_dimension_distances=True, 
                                        as_graph=False):
    means = gmm.means_
    memberships = gmm.predict(data)
    if connection_estimation_method in ['max_path_distance_diff', 'connecting_paths', 'mst']:
        if low_dimension_distances:
            pca = PCA(n_components=2)
            distance_matrix = squareform(pdist(pca.fit_transform(data), distance))
        else:
            distance_matrix = squareform(pdist(data, distance))
        weighted_graph = nx.Graph(distance_matrix)
    else:
        weighted_graph = None
    return get_metastable_connections(data, means, memberships, 
                                      method=connection_estimation_method, 
                                      weighted_graph=weighted_graph, min_paths=3,
                                      as_graph=as_graph)


def _path_memberships(paths, memberships):
    path_memberships = defaultdict(dict)
    for n1 in paths:
        for n2 in paths[n1]:
            path = paths[n1][n2]
            path_memberships[n1][n2] = np.unique(memberships[path])
    return path_memberships

def _max_path_distances(paths, distance_matrix):
    max_path_distances = defaultdict(dict)
    for n1 in paths:
        for n2 in paths[n1]:
            path = paths[n1][n2]
            dists = []
            if len(path) == 1:
                max_path_distances[n1][n2] = distance_matrix[n1, n2]
            else:
                for i in xrange(1, len(path)):
                    dists.append(distance_matrix[path[i-1], path[i]])
                max_path_distances[n1][n2] = max(dists)
    return max_path_distances

def _max_path_distribution(indices, max_path_distances, 
                           path_memberships, indices2=None,
                           max_memberships=2):
    dists = []
    if indices2 is None:
        indices2 = indices
    for i in indices:
        for j in indices2:
            if i != j:
                if len(path_memberships[i][j]) <= max_memberships:
                    dists.append(max_path_distances[i][j])
    return np.array(dists)

def get_metastable_connections(data, means, memberships, 
                               method='max_path_distance_diff',
                               weighted_graph=[], 
                               min_paths=3, effect_size_thresh=0.5, 
                               as_graph=False):
    edges = []
    if method == 'mst':
        data_mst_graph = nx.minimum_spanning_tree(weighted_graph)
        for m1_idx, mean in enumerate(means):
            for m2_idx, mean in enumerate(means):
                if m1_idx == m2_idx:
                    continue
                found = False
                for n1 in np.where(memberships == m1_idx)[0]:
                    for n2 in np.where(memberships == m2_idx)[0]:
                        if data_mst_graph.has_edge(n1, n2):
                            found = True
                            edges.append((m1_idx, m2_idx))
                            break
                    if found:
                        break

    if method == 'max_path_distance_diff':
        data_mst_graph = nx.minimum_spanning_tree(weighted_graph)
        data_shortest_paths, data_path_weighted_lengths = all_pairs_shortest_path_weighted(data_mst_graph)

        max_path_distances = _max_path_distances(data_shortest_paths, 
                                                nx.to_numpy_matrix(weighted_graph))

        path_memberships = _path_memberships(data_shortest_paths, memberships)
        
        membership_indices = {}
        max_path_distributions = {}
        for m_idx, m in enumerate(means):
            m_indices = np.where(memberships == m_idx)[0]
            membership_indices[m_idx] = m_indices

            m_distribution = _max_path_distribution(membership_indices[m_idx],
                                    max_path_distances, path_memberships,
                                    max_memberships=1)
            max_path_distributions[m_idx] = m_distribution

        for m1_idx, m1 in enumerate(means):
            for m2_idx, m2 in enumerate(means):
                if m1_idx == m2_idx:
                    continue
                m1_m2_distribution = _max_path_distribution(membership_indices[m1_idx],
                                    max_path_distances, path_memberships,
                                    indices2=membership_indices[m2_idx],
                                    max_memberships=2)

                d_1 = stats.cohens_d(m1_m2_distribution, 
                        max_path_distributions[m1_idx])

                d_2 = stats.cohens_d(m1_m2_distribution, 
                        max_path_distributions[m2_idx])

                if min(d_1, d_2) < effect_size_thresh:
                    edges.append((m1_idx, m2_idx))

    if method == 'connecting_paths':
        data_mst_graph = nx.minimum_spanning_tree(weighted_graph)
        data_shortest_paths = nx.all_pairs_dijkstra_path(data_mst_graph, 
                                                         weight='weight')

        for m1_idx, m1 in enumerate(means):
            for m2_idx, m2 in enumerate(means):
                if m1_idx == m2_idx:
                    continue
                num_total_paths = 0.
                num_connecting_paths = 0.
                for source in np.where(memberships == m1_idx)[0]:
                    for target in np.where(memberships == m2_idx)[0]:
                        found = True
                        for node in data_shortest_paths[source][target]:
                            if memberships[node] not in [m1_idx, m2_idx]:
                                found = False
                                break
                        if found:
                            num_connecting_paths += 1
                        num_total_paths += 1
                if num_connecting_paths >= min_paths and num_connecting_paths/num_total_paths > 0.5:
                    edges.append((m1_idx, m2_idx))

    if method == 'connecting_projections':
        for m1_idx, m1 in enumerate(means):
            for m2_idx, m2 in enumerate(means):
                if m1_idx == m2_idx:
                    continue
                diff_vec = (m2 - m1)
                diff_vec /= norm(diff_vec)
                for i in xrange(data.shape[0]):
                    dp = data[i, :]
                    proj_memberships = np.zeros([len(means)])
                    proj = m1 + np.dot(np.dot(diff_vec, dp - m1), diff_vec) 
                    if norm(m1) > norm(m2):
                        u_bound = norm(m1)
                        l_bound = norm(m2)
                    else:
                        u_bound= norm(m2)
                        l_bound = norm(m1)
                    if norm(proj) > u_bound or norm(proj) < l_bound:
                        continue
                    proj_memberships[memberships[i]] += 1
                if np.argmax(proj_memberships) in [m1_idx, m2_idx]:
                    edges.append((m1_idx, m2_idx))
                    
    if as_graph:
        g = nx.Graph()
        for n1, n2 in edges:
            g.add_edge(n1, n2, weight=weighted_graph[n1][n2]['weight'])
        return g
    else:
        return edges


