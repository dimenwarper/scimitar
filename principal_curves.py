import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

def _normalize_mapping(mapping):
    mapping -= mapping.min()
    mapping /= mapping.max()
    return mapping

def _map_data_to_principal_line(data_array):
    pca = PCA(n_components=2)
    pca.fit(data_array)
    mapping = np.zeros(data_array.shape[0])

    first_comp = pca.components_[0, :]
    vec = first_comp/np.linalg.norm(first_comp)
    
    mapping = np.dot(data_array, vec)
    return _normalize_mapping(mapping)

def _map_data_to_curve(data, curve):
        mapping_indices = np.zeros([data.shape[0]])
        distance_matrix = cdist(data, curve, metric='cosine')
        for i in xrange(data.shape[0]):
            mapping_indices[i] = np.argmin(distance_matrix[i, :])
        return mapping_indices

class PrincipalCurve():
    def __init__(self, n_iters=20, smooth_thresh=0.1, stop_thresh=0.001,
                 interval=np.arange(0, 1, 0.01)):
        self.n_iters = n_iters
        self.smooth_thresh = smooth_thresh
        self.stop_thresh = stop_thresh
        self.interval = interval

    def fit(self, data_array):
        curr_mapping = _map_data_to_principal_line(data_array)
        curr_error = np.inf
        for i in xrange(self.n_iters):    
            curr_curve = np.zeros([len(self.interval), data_array.shape[1]])
            for i, lam in enumerate(self.interval):
                chosen_indices = np.where((curr_mapping >= lam - self.smooth_thresh) & 
                                        (curr_mapping <= lam + self.smooth_thresh))[0]
                curr_curve[i, :] = data_array[chosen_indices, :].mean(axis=0)
            curr_curve = np.array(curr_curve)
            curr_mapping_indices = _map_data_to_curve(data_array, curr_curve)
            error = ((curr_curve[curr_mapping_indices.astype(int), :] - data_array)**2).sum()
            curr_mapping = _normalize_mapping(curr_mapping_indices)
            

            if curr_error - error < self.stop_thresh:
                break
            else:
                curr_error = error
        self.curve_ = curr_curve

    def predict(self, data_array):
        mapping = _map_data_to_curve(data_array, self.curve_)
        return _normalize_mapping(mapping)
