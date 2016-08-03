library('monocle')

run_monocle <- function(path){
  expr_matrix <- read.csv(paste('../insilico_data/', path,'_samples.tsv', sep=''), sep='\t')
  expr_matrix <- as.matrix(expr_matrix[,2:ncol(expr_matrix)])
  monocle_data <- newCellDataSet(t(expr_matrix - min(expr_matrix)))
  fData(monocle_data)$use_for_ordering = T
  monocle_data <- reduceDimension(monocle_data, use_irlba = F)
  monocle_data <- orderCells(monocle_data, num_paths = 1, reverse = F)
  write.graph(slot(monocle_data, 'minSpanningTree'), paste('monocle_tree_edge_list_', path, '.txt', sep=''), format='edgelist')
  write.table(monocle_data$Pseudotime, file=paste('monocle_pseudotime_', path, '.txt', sep=''), sep='\t')
  png(file=paste('monocle_run_', path, '.png', sep=''))
  plot_spanning_tree(monocle_data)
  dev.off()
}

for(i in 1:6){
  run_monocle(paste('intrinsic', i, sep=''))
}

for(i in 1:6){
  run_monocle(paste('extrinsic', i, sep=''))
}
