import pandas as pd
import numpy as np
import models
import plotting


def _print_with_options(print_str, options):
    if len(options) == 0:
        print_str += ' -- with default options'
        print print_str 
    else:
        print print_str
        print '='*len(print_str)
        print 'Options:'
        for key, val in options.iteritems():
            print '%s = %s' % (key, val)
    print '-'*len(print_str)

def preprocess(data_file, n_selected_genes=2000, log_transform=True, transpose=False):
    if transpose:
        data_df = pd.read_csv(data_file, sep='\t', comment='"').T
        data_df = data_df[(data_df.mean(axis=1) > 5).values]
        data_M = np.array(data_df.iloc[1:,:])
    else:
        data_df = pd.read_csv(data_file, sep='\t', comment='"')
        data_df = data_df[(data_df.mean(axis=1) > 5).values]
        data_M = np.array(data_df.iloc[:,1:])
    selected_genes = sorted(range(data_M.shape[1]), 
                            key=lambda i: data_M[:,i].std()**2, 
                            reverse=True)[:n_selected_genes]

    if log_transform:
        data_M = np.log(data_M[:, selected_genes] + 1)
    return data_df, data_M

def gmm_state_decomposition(data_file, n_states, preprocessing_options={}, decomposition_options={},
                            plotting_options={}):
    _print_with_options('Preprocessing data', preprocessing_options)
    data_df, data_M = preprocess(data_file, **preprocessing_options)

    _print_with_options('Performing state decomposition', decomposition_options)
    state_decomposition = models.get_gmm_state_decomposition(data_M, n_states, **decomposition_options)
    
    _print_with_options('Plotting', plotting_options)
    membership_colors, embedding = plotting.plot_states(data_M, state_decomposition, **plotting_options)
    return state_decomposition, data_M, membership_colors, embedding

