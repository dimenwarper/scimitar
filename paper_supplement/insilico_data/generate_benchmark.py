import numpy as np
from scimitar import simulation
from scimitar.simulation import BranchPointDict as bpd
from collections import OrderedDict

tree_struct = bpd(path_name='path1', children={0.7:bpd(path_name='path2',
                                         children={0.7:bpd(path_name='path3',
                                                           children={}),
                                                   0.71:bpd(path_name='path4',
                                                           children={}),
                                                   0.8:bpd(path_name='path5',
                                                           children={})}),
                                 0.2:bpd(path_name='path6',
                                         children={})})
param_dict = OrderedDict()
param_dict['intrinsic1'] = {'n_samples':100,
                       'degree':2,
                       'max_jitter_points':2,
                       'cov_fac':0.1, 'mean_fac':1}
param_dict['intrinsic2'] = {'n_samples':100,
                       'degree':2,
                       'max_jitter_points':2,
                       'cov_fac':0.5, 'mean_fac':1}
param_dict['intrinsic3'] = {'n_samples':100,
                       'degree':2,
                       'max_jitter_points':2,
                       'cov_fac':1, 'mean_fac':1}
param_dict['intrinsic4'] = {'n_samples':100,
                       'degree':2,
                       'max_jitter_points':2,
                       'cov_fac':3, 'mean_fac':1}
param_dict['intrinsic5'] = {'n_samples':100,
                       'degree':2,
                       'max_jitter_points':2,
                       'cov_fac':10, 'mean_fac':1}
param_dict['intrinsic6'] = {'n_samples':100,
                       'degree':2,
                       'max_jitter_points':2,
                       'cov_fac':20, 'mean_fac':1}

ndims = 10
seed = 23467
seed = 98732
for path_name, params in param_dict.iteritems():
    random_state = np.random.RandomState(seed)
    simulated_path = simulation.generate_path(path_name, ndims, random_state=random_state, **params)
    simulated_path.save(outprefix='./')
#simulated_tree = simulation.generate_tree(tree_struct, ndims, param_dict, random_state)
#simulated_tree.save(outprefix='./')



seed = 546789
random_state = np.random.RandomState(seed)

magnitudes = [0., 0.2, 0.4, 0.6, 0.8, 1.]
embedded_random_walks = simulation.simulate_embedded_random_walk_trajectories(nsteps=600, 
                                                      noise_mags=magnitudes, 
                                                      random_state=random_state)

i = 1
for mag in magnitudes:
    samples = embedded_random_walks[mag]
    samplefile = open('extrinsic%s_samples.tsv' % i, 'w')
    simulation.save_samples(samples, samplefile)
    i += 1
