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
 

def progression_association_f_test(data_array, transition_model, pseudotimes):
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

def progression_association_lr_test(data_array, means, variances, method='bootstrap', 
                       n_boot=1000):
    n_samples = data_array.shape[0]
    n_variables = data_array.shape[1]
    
    pvals = np.ones([n_variables])
    lrs = {}
    null_vars = {}
    null_means = {}
    for variable in xrange(n_variables):
            null_vars[variable] = data_array[:, variable].var()
            null_means[variable] = data_array[:, variable].mean() 
            
            log_probs = np.array([scipy.stats.norm.logpdf(data_array[:, variable], loc=means[i, variable], 
                                                     scale=variances[i, variable]**0.5)
                                           for i in xrange(means.shape[0])])
            log_p = np.array([np.argmax(log_probs[:, i]) for i in xrange(n_samples)])
            
            null_log_p = scipy.stats.norm.logpdf(data_array[:, variable], loc=null_means[variable], 
                                                 scale=null_vars[variable]**0.5)
            lrs[variable] = null_log_p.sum() - log_p.sum()
    if method == 'bootstrap':
        for variable in xrange(n_variables):
            boot_ratios = np.zeros([n_boot])
            for boot_iter in xrange(n_boot):
                boot_samples = scipy.stats.norm.rvs(loc=null_means[variable], scale=null_vars[variable]**0.5,
                                                    size=n_samples)
                null_boot_log_p = scipy.stats.norm.logpdf(boot_samples, loc=null_means[variable], 
                                                          scale=null_vars[variable]**0.5)
                
                boot_log_probs = np.array([scipy.stats.norm.logpdf(boot_samples, loc=means[i, variable], 
                                                     scale=variances[i, variable]**0.5)
                                           for i in xrange(means.shape[0])])
                boot_log_p = np.array([np.argmax(boot_log_probs[:, i]) for i in xrange(n_samples)])
                boot_ratios[boot_iter] = null_boot_log_p.sum() - boot_log_p.sum()
            boot_mean, boot_std = boot_ratios.mean(), boot_ratios.std()
            pval = scipy.stats.norm.pdf(lrs[variable], loc=boot_mean, scale=boot_std)
            if boot_mean < lrs[variable]:
                pvals[variable] = pval
            else:
                pvals[variable] = 1 - pval
    else:
        #TODO Should include the chisq approximation here
        raise ValueError('No method for progression association testing %s is available' % method)
    return pvals

