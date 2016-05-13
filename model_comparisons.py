from collections import defaultdict, Counter
from numpy.linalg import norm
from utils import create_state_index
import munkres
import numpy as np

def align_state_decompositions(state_decom_1, state_decom_2):
    state_mapping = {}
    mapping_scores = {}
    preserved_edges = []
    states_1 = state_decom_1.state_model.centroids
    states_2 = state_decom_2.state_model.centroids
    state_dist_matrix = np.zeros([len(states_1), len(states_2)])
    for i, s1 in enumerate(states_1):
        for j, s2 in enumerate(states_2):
            state_dist_matrix[i, j] = norm(s1 - s2)
    munkres_runner = munkres.Munkres()
    for idx1, idx2 in munkres_runner.compute(state_dist_matrix.tolist()):
        state_mapping[create_state_index(idx1)] = create_state_index(idx2)
    for s1, s2 in state_mapping.iteritems():
        mapping_scores[s1] = state_dist_matrix[s1.index, s2.index]
    for s1_1, s1_2 in state_decom_1.state_edges:
        if (state_mapping[s1_1], state_mapping[s1_2]) in state_decom_2.state_edges or \
           (state_mapping[s1_2], state_mapping[s1_1]) in state_decom_2.state_edges:
            preserved_edges.append((s1_1, s1_2))
    return state_mapping, preserved_edges, mapping_scores

def get_edge_fractions(state_decompositions):
    edge_fractions = defaultdict(float)
    num_decom = float(len(state_decompositions))
    ref_state_decom = state_decompositions[0]
    all_mapped_edges = []
    for s1, s2 in ref_state_decom.state_edges:
        if s1.index < s2.index:
            e1, e2 = s1, s2
        else:
            e1, e2 = s2, s1
        all_mapped_edges.append((e1, e2))

    for state_decom in state_decompositions[1:]:
        state_mapping, _, _ = align_state_decompositions(state_decom, ref_state_decom)
        for s1, s2 in state_decom.state_edges:
            if state_mapping[s1].index < state_mapping[s2].index:
                e1, e2 = state_mapping[s1], state_mapping[s2]
            else:
                e1, e2 = state_mapping[s2], state_mapping[s1]
            all_mapped_edges.append((e1, e2))
    edge_fractions = Counter(all_mapped_edges)
    for edge, count in edge_fractions.iteritems():
        edge_fractions[edge] = count/(num_decom*2)
    return edge_fractions


