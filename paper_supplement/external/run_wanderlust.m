for i=1:6
	data_s = tdfread(['../insilico_data/intrinsic', num2str(i), '_samples.tsv']);
	data_c = struct2cell(data_s);
	data_m = [data_c{2:end}];
	[graph_traj, lnn] = wanderlust( data_m, 'euclidean', 30, 5, 20, 1, 20, 1, [1]);
	save(['wanderlust_intrinsic', num2str(i), '_traj.txt'], 'graph_traj',  '-ascii');
end

for i=1:6
	data_s = tdfread(['../insilico_data/extrinsic', num2str(i), '_samples.tsv']);
	data_c = struct2cell(data_s);
	data_m = [data_c{2:end}];
	[graph_traj, lnn] = wanderlust( data_m, 'euclidean', 30, 5, 20, 1, 20   , 1, [1]);
	save(['wanderlust_extrinsic', num2str(i), '_traj.txt'], 'graph_traj',  '-ascii');
end

