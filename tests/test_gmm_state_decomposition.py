import pipelines
import plotting
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(1)
state_decom, data_array, membership_colors, embedding = pipelines.gmm_state_decomposition('test_data/trapnell.fpkm', 4, preprocessing_options={'transpose':True, 'n_selected_genes':100})
plt.savefig('tests/gmm_state_decomposition_plot.png')
states = list(state_decom.state_edges[0])
print 'Performing Transition modeling of states ' + str(states)
transition_model, model_trace, analyzed_indices = state_decom.get_transition_model(data_array, states=states, degree=2, n_iters=5)
plt.figure(2)
plotting.plot_transition_model(data_array[analyzed_indices,:], transition_model, embedding=embedding, colors=[membership_colors[i] for i in analyzed_indices])
plt.savefig('tests/transition_model.png')

