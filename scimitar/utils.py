from collections import defaultdict, namedtuple
import numpy as np
import networkx as nx

from scipy.spatial.distance import pdist, squareform

from . import settings

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

def get_data_delaunay_from_gmm(data, gmm, min_paths=3, distance='euclidean', as_graph=False):
    means = gmm.means_
    memberships = gmm.predict(data)
    distance_matrix = squareform(pdist(data, distance))
    weighted_graph = nx.Graph(distance_matrix)
    return get_data_delaunay(data, means, weighted_graph, memberships, min_paths=3, as_graph=as_graph)

def get_data_delaunay(data, means, weighted_graph, memberships, 
                      min_paths=3, as_graph=False):
    edges = []
    data_mst_graph = nx.minimum_spanning_tree(weighted_graph)
    data_shortest_paths, data_path_weighted_lengths = all_pairs_shortest_path_weighted(data_mst_graph)

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
            if num_connecting_paths >= min_paths and num_connecting_paths/num_total_paths > 0.1:
                edges.append((m1_idx, m2_idx))


            """
            diff_vec = (m2 - m1)
            diff_vec /= norm(diff_vec)
            memberships = []
            for m3_idx, m3 in enumerate(_means):
                if m3_idx in [m1_idx, m2_idx]:
                    memberships.append(m3_idx)
                    continue
                proj = m1 + np.dot(np.dot(diff_vec, m3 - m1), diff_vec) 
                if norm(m1) > norm(m2):
                    u_bound = norm(m1)
                    l_bound = norm(m2)
                else:
                    u_bound= norm(m2)
                    l_bound = norm(m1)
                if norm(proj) > u_bound or norm(proj) < l_bound:
                    continue
                if gmm.predict_proba(proj).max() > 0.01:
                    memberships.append(gmm.predict(proj)[0])
            if len(set(memberships)) ==  2:
                # Midpoint must significantly belong to one of the classes
                min_dist = float('inf')
                member_1, member_2 = None, None
                for member_idx_1 in [idx for idx, state in enumerate(data_memberships) if state == m1_idx]:
                    for member_idx_2 in [idx for idx, state in enumerate(data_memberships) if state == m2_idx]:
                        if min_dist > data_distance_M[member_idx_1, member_idx_2]:
                            min_dist = data_distance_M[member_idx_1, member_idx_2]
                            member_1 = data[member_idx_1,:]
                            member_2 = data[member_idx_2,:]
                if member_1 is not None:
                    p1 = gmm.predict_proba(member_2)[0][m1_idx]
                    p2 = gmm.predict_proba(member_1)[0][m2_idx]
                    if True or max(p1, p2) >= 1e-10:
                        edges.append((m1_idx, m2_idx))
            """
    if as_graph:
        g = nx.Graph()
        for n1, n2 in edges:
            g.add_edge(n1, n2, weight=weighted_graph[n1][n2]['weight'])
        return g
    else:
        return edges


