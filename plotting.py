import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import zscore
from sklearn.manifold import LocallyLinearEmbedding
import settings

def choose_2d_embedding(data, n_neighbors=None):
    if n_neighbors is None:
        n_neighbors = int(data.shape[0] * 0.8)
    method = 'standard'
    if data.shape[0] < 50:
        method = 'modified'
    embedding = LocallyLinearEmbedding(n_components=2, n_neighbors=n_neighbors,
                                       method=method)
    return embedding



def plot_states(data, state_decomposition, n_neighbors=None, plot_edges=True, 
                state_edges=[], edge_weights={}, embedding=None, 
                memberships=None):
    if embedding is None:
        embedding = choose_2d_embedding(data, n_neighbors=n_neighbors)
        transformed = embedding.fit_transform(data)
    else:
        transformed = embedding.transform(data)
    if memberships is None:
        memberships = state_decomposition.state_model.predict(data)
    if not plot_edges:
        plt.scatter(transformed[:, 0], transformed[:, 1], 
                   c=[settings.STATE_COLORS[i] for i in memberships], 
                   linewidth=0, alpha=0.6)
    else:
        fig, axarr = plt.subplots(2, sharex=True, sharey=True)
        axarr[0].scatter(transformed[:,0], transformed[:, 1], 
                   c=[settings.STATE_COLORS[i] for i in memberships],
                   linewidth=0, alpha=0.6)
        ordered_states = np.unique(memberships)
        state_centers = [embedding.transform(state_decomposition.state_model.centroids[s])[0] for s in ordered_states]
        sizes = [5000*np.log(1 + (memberships == state).sum()/float(len(memberships))) for state in ordered_states]
        X, Y = [center[0] for center in state_centers], [center[1] for center in state_centers]
        if len(state_edges) == 0:
            state_edges = state_decomposition.state_edges
        if len(edge_weights) == 0:
            edge_weights = state_decomposition.edge_weights
        for state1, state2 in state_edges:
            axarr[1].plot([X[state1.index], X[state2.index]], [Y[state1.index], Y[state2.index]],'k-', zorder=1, alpha=0.6)
            if (state1, state2) in edge_weights:
                midpoint_X = min(X[state1.index], X[state2.index]) + abs(X[state1.index] - X[state2.index])/2.
                midpoint_Y = min(Y[state1.index], Y[state2.index]) + abs(Y[state1.index] - Y[state2.index])/2.
                axarr[1].text(midpoint_X, midpoint_Y, '%s' % (int(edge_weights[(state1, state2)] * 100)) + '%', fontweight='bold', fontsize=12)

        
        axarr[1].scatter(X, Y, c=[settings.STATE_COLORS[i] for i in ordered_states], s=sizes, linewidth=0, zorder=2)
        return np.array([settings.STATE_COLORS[i] for i in memberships]), embedding

def plot_transition_model(data, transition_model, n_neighbors=None, embedding=None, colors='magenta',
                          plot_errors=True, timepoints=np.arange(0, 1, 0.02)):
    if embedding is None:
        embedding = choose_2d_embedding(data, n_neighbors=n_neighbors)
        transformed = embedding.fit_transform(data)
    else:
        transformed = embedding.transform(data)

    means = transition_model.mean(timepoints)
    interp_mean = embedding.transform(means)
    
    if plot_errors:
        samples = transition_model.sample_along_timepoints(timepoints, data.shape[0]*2,
                                                        low_dim_means=interp_mean,
                                                        even=True)
        interp_samples = embedding.transform(samples)

        D = pdist(interp_samples)
        bw = D.mean() - 1.5*D.std()
        X_kde = np.concatenate((interp_samples[:, 0], transformed[:, 0]))
        Y_kde = np.concatenate((interp_samples[:, 1], transformed[:, 1]))
        sns.kdeplot(X_kde, Y_kde, cut=1, n_levels=4, 
                    shade_lowest=False, shade=True, cmap='Greens')
    plt.scatter(transformed[:, 0], transformed[:, 1], color=colors, linewidth=0., alpha=0.5)
    plt.scatter(interp_mean[:, 0], interp_mean[:, 1], marker=(5, 2), color='black')
    for i in xrange(1, interp_mean.shape[0]):
        plt.plot([interp_mean[i-1, 0], interp_mean[i, 0]], 
                 [interp_mean[i-1, 1], interp_mean[i, 1]], color='black',
                 linewidth=3)
    return embedding

def plot_gene_clustermap_by_membership(data_array, memberships):
    groups = list(set(memberships))
    group_means = np.zeros([len(groups), data_array.shape[1]])
    for gi, group in enumerate(groups):
        group_means[gi, :] = data_array[memberships == group, :].mean(axis=0)
    cm = sns.clustermap((group_means - group_means.min(axis=0)).T, col_cluster=False)
    r = np.arange(len(groups)) + 0.5
    cm.ax_heatmap.set_xticks(r)
    cm.ax_heatmap.set_xticklabels(groups)
    cm.ax_heatmap.set_xlabel('Cell type')
    cm.ax_heatmap.set_ylabel('Gene')
    return cm

def plot_transition_clustermap(data_array, gene_names, pseudotimes, n_clusters=10, gradient=False):
    if gradient:
        data_to_plot = zscore(np.gradient(data_array)[1].T, axis=0)
        scale = None
        metric = 'seuclidean'
        row_linkage = linkage(pdist(abs(data_to_plot), metric=metric), method='complete')
    else:
        data_to_plot = data_array.T
        scale = 0
        metric = 'correlation'
        row_linkage = linkage(pdist(data_to_plot, metric=metric), method='complete')
    
    assignments = fcluster(row_linkage, n_clusters, criterion='maxclust')
    cm = sns.clustermap(data_to_plot, col_cluster=False, standard_scale=scale, 
                        yticklabels=gene_names, row_linkage=row_linkage,
                        row_colors=[settings.STATE_COLORS[i] for i in assignments])
    r = np.arange(10, data_array.shape[0], data_array.shape[0]/10)
    plt.setp(cm.ax_heatmap.get_yticklabels(), fontsize=5)
    cm.ax_heatmap.set_xticks(r)
    cm.ax_heatmap.set_xticklabels(['%.1f' % x for x in pseudotimes[r]])
    cm.ax_heatmap.set_xlabel('Pseudotime')
    cm.ax_heatmap.set_ylabel('Gene')
    
    gene_clusters = defaultdict(list)
    for i, cl in enumerate(assignments):
        gene_clusters[settings.STATE_COLORS[cl]].append(gene_names[i])
    return gene_clusters

def plot_transition_expression(gene_expression, model_means, model_variances,
                               pseudotimes, timepoints, color='black', ax=None):
    if ax is None:
        ax = plt.subplot(111)
    plt.scatter(pseudotimes, gene_expression, 
                color=color, alpha=0.6)
    ax.plot(timepoints, model_means, linewidth=2)
    ax.fill_between(timepoints, model_means - model_variances**0.5, 
                     model_means + model_variances**0.5, 
                     color='blue', alpha=0.2)

    plt.xlabel('Pseudotime')
    plt.ylabel('Expression')

def plot_transition_coexpression_expression(timepoints, degrees):
    ax = plt.subplot(211)
    ax.plot(timepoints, degrees, linewidth=5)
    plt.ylabel('Co-expression degree')

    plt.xlim(0, 1.)
    ax = plt.subplot(212)

