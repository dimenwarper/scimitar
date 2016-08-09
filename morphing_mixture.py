import numpy as np
import networkx as nx
import utils
from principal_curves import PrincipalCurve
from scipy.interpolate import UnivariateSpline
from sklearn import gaussian_process
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.mixture import GMM
from pyroconductor import corpcor, glasso
from scipy.spatial.distance import pdist, squareform
from scipy import stats
from numpy.linalg import norm
import warnings
warnings.simplefilter('ignore', np.RankWarning)

class MorphingGaussianMixture(object):
    def __init__(self, mean_funs=None, cholesky_funs=None, 
                 fit_type='spline', degree=3, cov_reg=0.01,
                 step_size=0.07,
                 reference_timepoints=np.arange(0, 1, 0.01), 
                 mean_coeffs=None, chol_coeffs=None,
                 cov_estimator='corpcor'):
        self.mean_coeffs = mean_coeffs
        self.chol_coeffs = chol_coeffs
        self.mean_funs = mean_funs
        if mean_funs is not None:
            self.ndimensions = len(mean_funs)
        self.cholesky_funs = cholesky_funs
        self.reference_timepoints = reference_timepoints
        self.fit_type = fit_type
        self.degree = degree
        self.cov_reg = cov_reg
        self.step_size = step_size
        self.cov_estimator = cov_estimator
    
    def _calculate_mean(self, t):
        return [self.mean_funs[i](t) for i in xrange(self.ndimensions)]

    def mean(self, timepoints):
        if type(timepoints) == float or type(timepoints) == np.float64:
            t = timepoints
            return self._calculate_mean(t)
        else:
            means = []
            for t in timepoints:
                means.append(self._calculate_mean(t))
            return np.array(means)
    
    def _calculate_covariance(self, t):
        chol_matrix = np.zeros([self.ndimensions, self.ndimensions])
        for indices, fun in self.cholesky_funs.iteritems():
            i, j = indices
            chol_matrix[i, j] = fun(t)
            chol_matrix[j, i] = chol_matrix[i, j]
        return np.dot(chol_matrix, chol_matrix.T)

    def covariance(self, timepoints):
        if type(timepoints) == float or type(timepoints) == np.float64:
            t = timepoints
            return self._calculate_covariance(t)
        else:
            cov_matrices = []
            for t in timepoints:
                cov_matrices.append(self._calculate_covariance(t))
            return np.array(cov_matrices)  

    def sample_along_timepoints(self, timepoints, sample_size, even=True, low_dim_means=None):
        samples = []
        means = self.mean(timepoints)
        covariances = self.covariance(timepoints)
        if even:
            if low_dim_means is None:
                means_for_distance_calc = means
            else:
                means_for_distance_calc = low_dim_means
            distances = np.zeros([len(timepoints)-1])
            for i in range(1, len(timepoints)):
                distances[i-1] = norm(means_for_distance_calc[i,:] - means_for_distance_calc[i-1,:])
            proportions = distances / distances.sum()
            sample_sizes = [max(1, int(sample_size*p)) for p in proportions]
        else:
            sample_sizes = [max(sample_size/(len(timepoints) - 1), 1)]*(len(timepoints) - 1)
        for i in xrange(len(timepoints) - 1):
            samples += np.random.multivariate_normal(means[i, :], covariances[i, :, :], sample_sizes[i]).tolist()
        return np.array(samples)

    def log_like(self, data_array):
        integration_range = np.arange(0, 1, 0.01)
        samples = np.array([stats.multivariate_normal(mean=self.mean(t), 
                                                     cov=self.covariance(t),
                                                     allow_singular=True).logpdf(data_array).tolist() 
                                                     for t in integration_range]).T

        res = 0.
        for i in xrange(samples.shape[0]):
            res += samples[i, :].max()
            #res += np.log(max(integral(samples[:, i], x=integration_range), 1e-100))
        return res

    def map_samples_to_pseudotime(self, samples,
                                  timepoints=np.arange(-0.5,1.5,0.01)):
        pseudotimes = np.zeros([samples.shape[0]])
        means = self.mean(timepoints)
        covs = self.covariance(timepoints)
        pt_probs = np.zeros([samples.shape[0], len(timepoints)])
        for j, t in enumerate(timepoints):
            pt_probs[:, j] = stats.multivariate_normal.logpdf(samples, mean=means[j,:], cov=covs[j,:,:], allow_singular=True)
        
        for i in xrange(samples.shape[0]):
            j = np.argmax(pt_probs[i, :])
            pseudotimes[i] = timepoints[j]
        return pseudotimes, pt_probs
    
    def refine(self, data_array, max_iter=10, **kwargs):
        current_transition_model = self
        print 'Initializing'
        prev_pseudotimes, prev_pt_probs = self.map_samples_to_pseudotime(data_array)
        current_pseudotimes = prev_pseudotimes
        R = 0
        for i in xrange(max_iter):
            print 'Iteration %s' % i 
            current_transition_model = morphing_mixture_from_pseudotime(data_array,
                                                            prev_pseudotimes, 
                                                            pt_probs=prev_pt_probs,
                                                            fit_type=self.fit_type, degree=self.degree, **kwargs)
            current_pseudotimes, curr_pt_probs = current_transition_model.map_samples_to_pseudotime(data_array)
            R = abs(stats.spearmanr(current_pseudotimes, prev_pseudotimes)[0])
            print 'R: %s' % R
            if R >= 0.9:
                print 'Converged!'
                break
            prev_pseudotimes = current_pseudotimes
        print 'Final R: %s' % R
        return current_transition_model, current_pseudotimes

    def fit(self, data_array):
        _mgm = morphing_gaussian_from_embedding(data_array, 
                                                cov_estimator=self.cov_estimator,
                                                fit_type=self.fit_type,
                                                degree=self.degree,
                                                step_size=self.step_size,
                                                cov_reg=self.cov_reg)
        self.__dict__.update(_mgm.__dict__)
        self.ndimensions = data_array.shape[1]

    
def get_1d_ordering(data_array, means, covariances, covariance_type):
    """
    gmm = GMM(covariance_type=covariance_type)
    gmm.means_ = means
    gmm.covars_ = covariances
    gmm.weights_ = np.array([1./len(means)]*len(means))
    metastable_connections = utils.get_metastable_connections_from_gmm(data_array, gmm, 
                                                     distance='cosine',
                                                     as_graph=True)
    """
    distance_matrix = squareform(pdist(means, 'cosine'))
    g = nx.Graph(distance_matrix)
    metastable_connections = nx.minimum_spanning_tree(g, weight='weight')
    paths, path_weighted_lengths = utils.all_pairs_shortest_path_weighted(metastable_connections)
    max_len = 0
    for n1 in paths:
        for n2 in paths[n1]:
            if max_len < path_weighted_lengths[n1][n2]:
                max_len = path_weighted_lengths[n1][n2]
                source = n1
                target = n2
    max_len = 0
    for path in nx.all_simple_paths(metastable_connections, source, target):
        if len(path) > max_len:
            max_len = len(path)
            diameter_path = path
            if max_len == metastable_connections.number_of_nodes():
                break
    print max_len, diameter_path
    t = 0.
    timepoints = [t]
    order = [source]
    diameter_path = paths[source][target]
    for i in xrange(1, len(diameter_path)):
        t += metastable_connections[diameter_path[i-1]][diameter_path[i]]['weight']
        timepoints.append(t)
        order.append(diameter_path[i])
    timepoints = np.array(timepoints).T
    timepoints /= timepoints.max()
    return order, timepoints

def state_interpolation(data_array, means, covariances, 
                        timepoints=None, fit_type='spline',
                        degree=3, covariance_type='full'):
    if timepoints is None:
        order, timepoints = get_1d_ordering(data_array, means, 
                                            covariances, covariance_type)
    else:
        order, timepoints = zip(*sorted(enumerate(timepoints), key=lambda x: x[1]))
    sorted_means = np.array([means[i, :] for i in order])
    if covariance_type in ['diag', 'spherical']:
        sorted_covariances = np.array([np.diag(covariances[i, :]) for i in order])
    else:
        sorted_covariances = np.array([covariances[i, :, :] for i in order])

    sorted_chols = np.array([np.linalg.cholesky(c).tolist() for c in sorted_covariances])
    ndimensions = means[0, :].shape[0]
    
    mean_funs = {}
    chol_funs = {}
    
    if 'gaussian_process' in fit_type:
        timepoints = np.array([timepoints]).T
    else:
        dim = len(means[0,:])
        mean_coeffs = np.zeros([degree + 1, dim])
        chol_coeffs = np.zeros([degree + 1, dim, dim])
    
    for i in xrange(ndimensions):
        if fit_type == 'polynomial':
            coeffs = np.polyfit(timepoints, sorted_means[:, i], degree, w=sorted_covariances[:, i, i])
            mean_coeffs[:, i] = coeffs
            mean_funs[i] = lambda t, coeffs=coeffs: np.polyval(coeffs, t)
        elif fit_type == 'spline':
            spl = UnivariateSpline(timepoints, sorted_means[:, i], k=degree, w=sorted_covariances[:, i, i])
            mean_funs[i] = lambda t, fun=spl: fun(t)
             
        elif 'gaussian_process' in fit_type:
            if 'naive' in fit_type:
                gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
            else:
                nugget = max(0.01, sorted_means[:, i].std()**2)
                nugget = nugget/(sorted_means[:, i]**2)
                gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1, 
                                                      nugget=nugget)
            gp.fit(timepoints, sorted_means[:, i])
            mean_funs[i] = lambda t, gp=gp: gp.predict(t)[0]

        else:
            raise ValueError('Unvalid fit type %s' % fit_type)

        for j in xrange(i, ndimensions):
            if covariance_type == 'full' or i == j:
                if fit_type == 'polynomial':
                    coeffs = np.polyfit(timepoints, sorted_chols[:, j, i], degree)
                    chol_coeffs[:, j, i] = coeffs
                    chol_funs[(j, i)] = lambda t, coeffs=coeffs: np.polyval(coeffs, t)
                elif fit_type == 'spline':
                    spl = UnivariateSpline(timepoints, sorted_chols[:, j, i], k=degree)
                    chol_funs[(j, i)] = lambda t, fun=spl: fun(t)
                elif 'gaussian_process' in fit_type:
                    if 'naive' in fit_type:
                        gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
                    else:
                        nugget = max(0.01, sorted_chols[:, j, i].std()**2)
                        nugget = nugget/(sorted_chols[:, j, i]**2)
                        gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1, nugget=nugget)
                    gp.fit(timepoints, sorted_chols[:, j, i])
                    chol_funs[(j, i)] = lambda t, gp=gp: gp.predict(t)[0]
                else:
                    raise ValueError('Unvalid fit type %s' % fit_type)
            else:
                chol_funs[(j, i)] = lambda t: 0.
    if fit_type == 'polynomial':
        return MorphingGaussianMixture(mean_funs, chol_funs, fit_type, degree,
                                       reference_timepoints=timepoints, 
                                       mean_coeffs=mean_coeffs, chol_coeffs=chol_coeffs)
    else:
        return MorphingGaussianMixture(mean_funs, chol_funs, fit_type, degree,
                                       reference_timepoints=timepoints)


def morphing_mixture_from_pseudotime(data_array, pseudotimes, pt_probs=None, 
                                     step_size=0.07,
                                     fit_type='spline', degree=3,
                                     cov_estimator='corpcor', cov_reg=None):
    min_pt = pseudotimes.min()
    max_pt = pseudotimes.max()
    means, covariances = [], []
    timepoints = []
    for tp in np.arange(min_pt, max_pt, step_size):
        weights = np.exp(-0.5*((pseudotimes - tp)/(step_size))**2)
        weights = np.reshape(weights, [len(weights), 1])
        weights /= weights.sum()
        weighted_data = weights * data_array
        means.append(weighted_data.sum(axis=0))
        if cov_estimator == 'identity':
            covariances.append(np.eye(data_array.shape[1]))
        elif cov_estimator == 'diag':
            covariances.append(np.diag(weighted_data.std(axis=0)**2))
        elif cov_estimator == 'sample':
            covariances.append(np.cov(weighted_data.T))
        elif cov_estimator == 'glasso':
            if cov_reg is None and tp == min_pt:
                l1_trace = np.linspace(np.percentile(abs(weighted_data), 5), 
                                       np.percentile(abs(weighted_data), 50), 10)
                rhos = np.zeros([len(l1_trace)])
                loglikes = np.zeros([len(l1_trace)])
                for i, rho in enumerate(l1_trace):
                    print 'Doing glasso for rho=%s' % rho
                    res = glasso.glasso(np.cov(weighted_data.T), rho)
                    rhos[i] = rho
                    loglikes[i] = res[2]
                cov_reg = rhos[np.argmax(loglikes)]
            res = glasso.glasso(np.cov(weighted_data.T), cov_reg)
            covariances.append(res[0])
        elif cov_estimator == 'corpcor':
            if cov_reg is None:
                covariances.append(np.copy(corpcor.cov_shrink(data_array, weights=weights)))
            else:
                covariances.append(np.copy(corpcor.cov_shrink(data_array, weights=weights, **{'lambda':cov_reg})))
        else:
            raise ValueError('Covariance estimator %s not supported' % cov_estimator)
        timepoints.append(tp)

    '''
    min_win_size = 15
    for window_start in np.arange(min_pt, max_pt, step_size):
        window_samples = []
        step = 1
        while len(window_samples) < min_win_size and step * step_size < max_pt:
            window_samples = np.where((pseudotimes >= window_start) &
                                      (pseudotimes <= step * step_size))[0]
            step += 1
        
        if len(window_samples) < min_win_size:
            sorted_pseudotimes = sorted(enumerate(pseudotimes), key=lambda x:x[1])
            window_samples = [i for i, pt in sorted_pseudotimes[-min_win_size:]]
        print 'window size %s' % len(window_samples)
        means.append(data_array[window_samples, :].mean(axis=0))
        if cov_estimator == 'identity':
            covariances.append(np.eye(data_array.shape[1]))
        else:
            #cov_estimator.fit(data_array[window_samples, :])
            #covariances.append(np.copy(cov_estimator.covariance_))
            covariances.append(np.copy(corpcor.cov_shrink(data_array[window_samples, :])))
        timepoints.append(window_start)
    '''
    means = np.array(means)
    covariances = np.array(covariances)
    timepoints = np.array(timepoints)/max(timepoints)
    return state_interpolation(data_array, means, covariances, 
                               fit_type=fit_type, degree=degree,
                               timepoints=timepoints)

class Sampler(object):
    def next(self):
        raise NotImplementedError('Sampler.next not implemented')

    def current(self):
        raise NotImplementedError('Sampler.current not implemented')


class PrincipalCurveSampler(Sampler):
    def __init__(self, data_array, fit_type='spline', degree=3):
        self.degree = degree
        self.data = data_array
        self.fit_type = fit_type
        self.current_model = self.next()

    def next(self):
        smooth_thresh = np.random.rand()*0.3 + 0.1
        pcurve = PrincipalCurve(smooth_thresh=smooth_thresh, 
                                interval=np.arange(0, 1, 0.1))
        pcurve.fit(self.data)
        covariances = np.ones([len(pcurve.curve_), self.data.shape[1]])
        self.current_model = state_interpolation(self.data, pcurve.curve_, covariances, 
                               fit_type=self.fit_type, degree=self.degree,
                               timepoints=pcurve.interval,
                               covariance_type='diag')
        return self.current_model

        self.current_model = morphing_mixture_from_pseudotime(self.data, 
                mapping, fit_type=self.fit_type, degree=self.degree)
        return self.current_model

    def current(self):
        return self.current_model

class GMMInterpolationSampler(Sampler):
    def __init__(self, data_array, fit_type='spline', n_components=3, degree=3):
        Sampler.__init__(self)
        self.degree = degree
        self.data = data_array
        self.fit_type = fit_type
        if self.degree is None:
            self._min_k = 3
        else:
            self._min_k = self.degree + 1
        self.current_k = max(self._min_k, n_components)

        self.current_model = self.next()


    def next(self):
        if np.random.rand() > 0.5:
            if self.current_k > self._min_k:
                self.current_k -= 1
        else:
            self.current_k += 1
        gmm = GMM(n_components=self.current_k,  covariance_type='diag', params='m')
        gmm.fit(self.data)
        self.current_model = state_interpolation(self.data, gmm.means_, gmm.covars_, 
                                                 degree=self.degree, fit_type=self.fit_type,
                                                 covariance_type='diag')
        return self.current_model

    def current(self):
        return self.current_model

def morphing_gaussian_from_embedding(data_array, n_neighbors=None, 
                                     cov_estimator='corpcor', **kwargs):
    if n_neighbors is None:
        n_neighbors = int(data_array.shape[0] * 0.5)
    embedding = LocallyLinearEmbedding(n_components=1, n_neighbors=n_neighbors)
    u, s, v = np.linalg.svd(data_array, full_matrices=1)
    l = 2
    denoised_data_array = np.dot(u[:, :l], np.dot(np.diag(s[:l]), v[:l, :]))
    pseudotimes = embedding.fit_transform(denoised_data_array)

    pseudotimes -= pseudotimes.min()
    pseudotimes /= pseudotimes.max()
    mgm = morphing_mixture_from_pseudotime(data_array, 
                pseudotimes, cov_estimator=cov_estimator, **kwargs)
    return mgm


def sample_morphing_gaussian_mixtures(data_array, method='principal_curve', fit_type='spline', n_iters=1000, degree=3, n_components=5):
    print 'Initializing'
    if method == 'principal_curve':
        sampler = PrincipalCurveSampler(data_array, fit_type=fit_type, degree=degree)
    elif method == 'gaussian_mixture':
        sampler = GMMInterpolationSampler(data_array, fit_type=fit_type, degree=degree, n_components=n_components)
    else:
        raise ValueError('Invalid method for sampling morphing mixture %s!' % method)
    curr_log_p = sampler.current().log_like(data_array)
    trace = []
    trace.append({'model':sampler.current(), 'log_p':curr_log_p})
    model_opt = sampler.current()
    log_p_opt = curr_log_p
    print 'Starting sampling'
    for iter in range(n_iters):
        print 'At iteration %s' % (iter)
        mgm = sampler.next()
        new_log_p = mgm.log_like(data_array)
        if new_log_p > log_p_opt:
                log_p_opt = new_log_p
                model_opt = mgm
        """
        # For MCMC sampling, but not used for now
        alpha = new_log_p - curr_log_p
        if True or alpha >= 0 or np.random.rand() <= np.exp(alpha):
            print 'Accepted'
            curr_log_p = new_log_p
            print curr_log_p
        """
        trace.append({'model':mgm, 'log_p':new_log_p})
    return trace, model_opt, log_p_opt

def goodness_of_fit(data_array, pseudotimes, means, covs, timepoints, degree):
    indices = []
    for pt in pseudotimes:
        indices.append(np.argmin((pt - timepoints)**2))
    numer = (data_array - means[indices, :])**2
    ndeg = data_array.shape[0] - 1 - 2*degree
    denom = np.array([np.diag(covs[i, :, :]) for i in indices])*ndeg
    chi_sq = numer/denom
    return chi_sq.sum(axis=0)


