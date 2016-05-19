import numpy as np
import scipy.stats
import pyroconductor as pyr

def detect_change_tp(timepoints, means, covariances):
    change_tp = []
    for gene in xrange(means.shape[1]):
        m0 = means[0, gene]
        s0 = covariances[0, gene, gene]**0.5
        found = False
        for i, t in enumerate(timepoints):
            s = covariances[i, gene, gene]**0.5
            m = means[i, gene]
            factor = 1
            if m0 + factor*s0 < m - factor*s or m0 - factor*s0 > m + factor*s:
                found = True
                change_tp.append(t)
                break
        if not found:
            change_tp.append(np.inf)
    return np.array(change_tp)

def p_adjust(pvals, threshold, correction_method='BH'):
    if correction_method != 'none' and correction_method is not None:
        corr_pvals = np.array(pyr.r['p.adjust'](pvals, method=correction_method))
    genes_sorted_by_pval = [i for i, pval in 
                            sorted(enumerate(pvals), key=lambda x:x[1])
                            if pval < threshold]
    return corr_pvals, genes_sorted_by_pval
 

def transition_f_test(data_array, transition_model, pseudotimes,
                      correction_method='BH', threshold=0.):
    n_samples = data_array.shape[0]
    
    fit_means = transition_model.mean(pseudotimes)
    pvals = np.ones([data_array.shape[1]])
    d1 = n_samples - 1 - (n_samples - transition_model.degree), 
    d2 = n_samples - transition_model.degree
    for gene in xrange(data_array.shape[1]):
        ss_fit = ((data_array[:, gene] - fit_means[:, gene])**2).sum()

        mu_0 = data_array[:, gene].mean()
        ss_0 = ((data_array[:, gene] - mu_0)**2).sum()
        
        F = (ss_0 - ss_fit/d1)/(ss_fit/d2)
        pvals[gene] = scipy.stats.f.sf(F, d1, d2)
    return pvals

