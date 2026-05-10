# figures for the paper
python plot_results.py 21 --save_name figures/sys_mat_spec_dist_vs_density_linear_correlation.png --system_matrix_type adjacency neg_laplacian --ylabel "System matrix\nspectral distance\nversus density" --thresholds 0.8 0.95 --threshold_line_mode horizontal --no_averaging --time_horizons 0.01 0.1 1 10
python plot_results.py 25 27 --save_name figures/sys_mat_spec_dist_vs_gramian_spec_dist.png --system_matrix_type adjacency neg_laplacian --ylabel "System matrix spectral distance\nvs. Gramian spectral distance" --thresholds 0.8 0.95 --threshold_line_mode horizontal --no_averaging --time_horizons 0.01 0.1 1 10
python plot_results.py 33 35 --save_name figures/density_vs_gramian_spec_dist.png --system_matrix_type adjacency neg_laplacian --ylabel "Density versus controllability\nGramian spectral distance" --thresholds 0.8 0.95 --threshold_line_mode horizontal --no_averaging --time_horizons 0.01 0.1 1 10

