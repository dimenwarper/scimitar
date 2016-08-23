import numpy as np
import networkx as nx
from scipy.interpolate import UnivariateSpline
from collections import namedtuple

# path_name: str, children: dict of keys:timepoints, values:BranchPointDict
BranchPointDict = namedtuple('BranchPointDict', ['path_name', 'children'])

def save_samples(samples, samplefile):
    ndims = samples.shape[1]
    samplefile.write('Sample\t' + 
                    '\t'.join(['Gene%s' % i for i in xrange(ndims)]) + '\n')
    for i in xrange(samples.shape[0]):
        samplefile.write('Sample%s\t' % i + 
                        '\t'.join([str(x) for x in samples[i, :]]) + '\n')
    samplefile.close()

class SimulatedPath(object):
    def __init__(self, name='', samples=[], mean_fun={}, basis_cov_fun={},
                 timepoints=[]):
        self.name = name
        self.samples = samples
        self.mean_fun = mean_fun
        self.basis_cov_fun = basis_cov_fun
        self.interval = np.arange(0, 1, 0.05)
        self.timepoints = timepoints
    
    @property
    def covariances(self):
        if getattr(self, '_covariances', None) is None:
            ndims = self.samples.shape[1]
            self._covariances = np.zeros([len(self.interval), ndims, ndims])
            for ti, tp in enumerate(self.interval):
                basis_cov = self.basis_covs[ti, :, :]
                self._covariances[ti, :, :] = np.dot(basis_cov, basis_cov.T)
        return self._covariances
    
    def basis_cov(self, timepoints):
        indices = self._map_to_interval(timepoints)
        return self.basis_covs[indices, :, :]
    
    @property
    def basis_covs(self):
        if getattr(self, '_basis_covs', None) is None:
            ndims = self.samples.shape[1]
            self._basis_covs = np.zeros([len(self.interval), ndims, ndims])
            for ti, t in enumerate(self.interval):
                for i in xrange(ndims):
                    for j in xrange(i + 1):
                        self._basis_covs[ti, i, j] = self.basis_cov_fun[i, j](t)
        return self._basis_covs

    
    @property
    def means(self):
        if getattr(self, '_means', None) is None:
            ndims = self.samples.shape[1]
            self._means = np.array([[self.mean_fun[i](t) for i in xrange(ndims)] for t in self.interval])
        return self._means
    
    def _map_to_interval(self, timepoints):
        indices = []
        for tp in timepoints:
            indices.append(np.argmin(abs(self.interval - tp)))
        return indices

    def mean(self, timepoints):
        indices = self._map_to_interval(timepoints)
        return self.means[indices, :]
    
    def covariance(self, timepoints):
        indices = self._map_to_interval(timepoints)
        return self.covariances[indices, :, :]

    def save(self, outprefix=''):
        ndims = self.samples.shape[1]
        samplefile = open('%s%s_samples.tsv' % (outprefix, self.name), 'w')
        save_samples(self.samples, samplefile)
        np.save('%s%s_covariances' % (outprefix, self.name), self.covariances)
        np.save('%s%s_means' % (outprefix, self.name), self.means)
        np.save('%s%s_timepoints' % (outprefix, self.name), self.timepoints)
        np.save('%s%s_samples' % (outprefix, self.name), self.samples)

class SimulatedTree(object):

    def __init__(self, simulated_path, subtrees={}):
        self.simulated_path = simulated_path
        self.subtrees = subtrees

    def add_subtrees(self, timepoints, simulated_trees):
        for i, timepoint in enumerate(timepoints):
            self.subtrees[timepoint] = simulated_trees[i]

    def save(self, outprefix=''):
        self.simulated_path.save(outprefix=outprefix)
        if len(self.subtrees) > 0:
            for tree in self.subtrees.values():
                tree.save(outprefix=outprefix)


def random_curve(degree, max_jitter_points,
                      magnitude, start_point, 
                      end_point, enforce_positive=False,
                      random_state=None):
    if start_point == end_point:
        return lambda x, point=start_point: point
    n_jitter_points = random_state.randint(degree - 1, high=(max_jitter_points + 1))
    curve_points = np.zeros([n_jitter_points + 2])
    curve_points[0], curve_points[-1] = start_point, end_point
    timepoints = np.arange(0., 1., 1./len(curve_points))

    line_points = (end_point - start_point)*timepoints + start_point 
    jitter_fac = magnitude*abs(end_point - start_point)
    if enforce_positive:
        curve_points[1:-1] = line_points[1:-1] + jitter_fac*random_state.rand(n_jitter_points)
    else:
        curve_points[1:-1] = line_points[1:-1] + jitter_fac*random_state.randn(n_jitter_points)
    coeffs = np.polyfit(timepoints, curve_points, degree) 
    return lambda x, coeffs=coeffs, fac=magnitude: fac*np.polyval(coeffs, x)
    
    #spl = UnivariateSpline(timepoints, curve_points, k=degree)
    return lambda x, fun=spl: fun(x)


def generate_basis_covariance_function(ndims, degree, max_jitter_points, 
                                       cov_fac, start_basis_cov, 
                                       end_basis_cov,
                                       random_state=None):
    basis_cov_fun = {}
        
    for i in xrange(ndims):
        for j in xrange(0, i + 1):
            enforce_positive = i == j
            basis_cov_fun[i, j] = random_curve(degree, max_jitter_points,
                                                    cov_fac, 
                                                    start_basis_cov[i, j],
                                                    end_basis_cov[i, j],
                                                    enforce_positive=enforce_positive,
                                                    random_state=random_state)
    return basis_cov_fun

def generate_mean(ndims, degree, max_jitter_points, mean_fac, 
                  start_mean, end_mean, random_state=None):
    mean_fun = {}
    for i in xrange(ndims):
        mean_fun[i] = random_curve(degree, max_jitter_points, mean_fac,
                                        start_mean[i], end_mean[i],
                                        random_state=random_state)
    return mean_fun

def generate_basis_cov(ndims, cov_fac, cov_connectivity, 
                       cov_sim_prob, random_state):
    network_seed = random_state.randint(0, high=100000)
    G = nx.watts_strogatz_graph(ndims, cov_connectivity, cov_sim_prob,
                                seed=network_seed)
    cov_matrix_mask = nx.adjacency_matrix(G) != 1
    cov_matrix = random_state.randn(ndims, ndims) * cov_fac

    cov_matrix[cov_matrix_mask.todense()] = 0.
    diag = np.diag(cov_matrix)**2
    cov_matrix[np.tril_indices(ndims)] = cov_matrix[np.triu_indices(ndims)]
    #cov_matrix = np.dot(cov_matrix, cov_matrix.T)
    print 'Non zero %s of %s' % ((cov_matrix != 0).sum(), ndims**2/2)
    added_fac = 2
    pos_def = False
    while not pos_def:
        try:
            cov_matrix[np.diag_indices(ndims)] = diag + added_fac*cov_fac
            return np.linalg.cholesky(cov_matrix)
        except np.linalg.linalg.LinAlgError:
            added_fac += 1

def generate_tree(branch_point_dict, ndims, param_dict, random_state):
    path = generate_path(branch_point_dict.path_name, ndims,
                         **param_dict[branch_point_dict.path_name])
    tree = SimulatedTree(path, subtrees={})
    subtrees = []
    timepoints = []
    for timepoint, bpd in branch_point_dict.children.iteritems():
        start_mean = path.mean([timepoint])[0, :] 
        end_mean = path.mean([1])[0, :]

        mask = random_state.randint(0, high=ndims, size=random_state.randint(1, high=ndims/2))
        param_dict[bpd.path_name]['start_mean'] = start_mean
        param_dict[bpd.path_name]['end_mean'] = end_mean
        perturb = random_state.randn(len(mask))*np.linalg.norm(end_mean - start_mean)*0.3
        perturb_sign = perturb/abs(perturb)

        param_dict[bpd.path_name]['end_mean'][mask] += perturb_sign*np.linalg.norm(end_mean - start_mean)*0.1
        param_dict[bpd.path_name]['end_mean'][mask] += perturb
        
        param_dict[bpd.path_name]['start_basis_cov'] = path.basis_cov([timepoint])[0, :, :]
        
        subtree = generate_tree(bpd, ndims, param_dict, random_state)
        subtrees.append(subtree)
    timepoints = branch_point_dict.children.keys()
    tree.add_subtrees(timepoints, subtrees)
    return tree


def generate_path(name, ndims, degree=2, n_samples=100, max_jitter_points=3,
                  mean_fac=1, cov_fac=0.2,
                  start_mean=None, start_basis_cov=None,
                  end_mean=None, end_basis_cov=None,
                  random_state=None, cov_sim_prob=0.2,
                  cov_connectivity=None, cell_density='uniform'):
    if cov_connectivity is None:
        cov_connectivity = max(3, int(ndims/10))
    if random_state is None:
        random_state = np.random
    if start_mean is None:
        start_mean = random_state.randn(ndims)
    if start_basis_cov is None:
        start_basis_cov = generate_basis_cov(ndims, cov_fac, cov_connectivity,
                                             cov_sim_prob, random_state)
    if end_mean is None:
        direction = np.zeros([ndims])
        mask = random_state.randint(0, high=ndims, size=random_state.randint(1, high=ndims/2))
        mask = range(ndims/2)
        direction[mask] = mean_fac * random_state.randn(len(mask))
        end_mean = start_mean + 10*direction
        #end_mean = random_state.randn(ndims)
    if end_basis_cov is None:
        end_basis_cov = generate_basis_cov(ndims, cov_fac, cov_connectivity,
                                           cov_sim_prob, random_state)

    mean_fun = generate_mean(ndims, degree, max_jitter_points, mean_fac,
                             start_mean, end_mean, random_state=random_state)

    basis_cov_fun = generate_basis_covariance_function(ndims, degree, 
                                                       max_jitter_points, 
                                                       cov_fac,
                                                       start_basis_cov, 
                                                       end_basis_cov, 
                                                       random_state=random_state)

    timepoints = np.zeros([n_samples])
    for i in xrange(n_samples):
        if cell_density == 'uniform':
            timepoints[i] = random_state.rand()
        elif cell_density == 'metastable':
            state = random_state.choice(['start_state', 'end_state'])
            if state == 'start_state':
                timepoints[i] = min(1, random_state.exponential(scale=0.2))
            if state == 'end_state':
                timepoints[i] = max(0, 1 - random_state.exponential(scale=0.2))
        else:
            raise ValueError('cell_density should be "uniform" or "metastable", received %s' % cell_density)

    samples = np.zeros([len(timepoints), ndims])
    
    for ti, t in enumerate(timepoints):
        mean = np.array([mean_fun[i](t) for i in xrange(ndims)])
        basis_cov = np.zeros([ndims, ndims])
        for i in xrange(ndims):
            for j in xrange(0, i + 1):
                basis_cov[i, j] = basis_cov_fun[i, j](t)
        cov = np.dot(basis_cov, basis_cov.T)
        samples[ti, :] = random_state.multivariate_normal(mean, cov)
    path = SimulatedPath(name=name, samples=samples, timepoints=timepoints, 
                         mean_fun=mean_fun, basis_cov_fun=basis_cov_fun)
    return path

def random_walk_trajectory(ndims, nsteps, random_state):
    curr_point = np.zeros([ndims]) + 1
    trajectory = np.zeros([nsteps + 1, ndims])
    trajectory[0, :] = curr_point
    for i in xrange(1, nsteps + 1):
        curr_point += random_state.rand(ndims)
        trajectory[i, :] = curr_point
    return trajectory

def simulate_embedded_random_walk_trajectories(ndims=3, n_extra_dims=7, nsteps=10000,
                                         noise_mean=0, noise_mags=[0.],
                                         random_state=None):
    if random_state is None:
        random_state = np.random
    trajectory = random_walk_trajectory(ndims, nsteps, random_state)
    trajectory_range = trajectory.max() - trajectory.min()
    embedded_trajectories = {}
    for mag in noise_mags:
        scale = mag * trajectory_range
        extra_dims = random_state.randn(trajectory.shape[0], n_extra_dims) * scale
        traj = np.hstack([trajectory, extra_dims])
        embedded_trajectories[mag] = traj + random_state.randn(*traj.shape)

    return embedded_trajectories
