from collections import defaultdict, Counter
from numpy.linalg import norm
import munkres
import numpy as np

from . import utils

def align_metastable_graphs(metastable_graph_1, metastable_graph_2):
    state_mapping = {}
    mapping_scores = {}
    preserved_edges = []
    states_1 = metastable_graph_1.state_model.centroids
    states_2 = metastable_graph_2.state_model.centroids
    state_dist_matrix = np.zeros([len(states_1), len(states_2)])
    for i, s1 in enumerate(states_1):
        for j, s2 in enumerate(states_2):
            state_dist_matrix[i, j] = norm(s1 - s2)
    munkres_runner = munkres.Munkres()
    for idx1, idx2 in munkres_runner.compute(state_dist_matrix.tolist()):
        state_mapping[utils.create_state_index(idx1)] = utils.create_state_index(idx2)
    for s1, s2 in state_mapping.iteritems():
        mapping_scores[s1] = state_dist_matrix[s1.index, s2.index]
    for s1_1, s1_2 in metastable_graph_1.state_edges:
        if (state_mapping[s1_1], state_mapping[s1_2]) in metastable_graph_2.state_edges or \
           (state_mapping[s1_2], state_mapping[s1_1]) in metastable_graph_2.state_edges:
            preserved_edges.append((s1_1, s1_2))
    return state_mapping, preserved_edges, mapping_scores

def get_edge_fractions(metastable_graphs):
    edge_fractions = defaultdict(float)
    num_graphs = float(len(metastable_graphs))
    ref_metastable_graph = metastable_graphs[0]
    all_mapped_edges = []
    for s1, s2 in ref_metastable_graph.state_edges:
        if s1.index < s2.index:
            e1, e2 = s1, s2
        else:
            e1, e2 = s2, s1
        all_mapped_edges.append((e1, e2))

    for metastable_graph in metastable_graphs[1:]:
        state_mapping, _, _ = align_metastable_graphs(metastable_graph, ref_metastable_graph)
        for s1, s2 in metastable_graph.state_edges:
            if state_mapping[s1].index < state_mapping[s2].index:
                e1, e2 = state_mapping[s1], state_mapping[s2]
            else:
                e1, e2 = state_mapping[s2], state_mapping[s1]
            all_mapped_edges.append((e1, e2))
    edge_fractions = Counter(all_mapped_edges)
    for edge, count in edge_fractions.iteritems():
        edge_fractions[edge] = count/(num_graphs*2)
    return edge_fractions


