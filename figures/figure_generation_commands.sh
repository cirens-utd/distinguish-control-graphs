# figures for the paper
python plot_results.py 30 33 --save_name figures/sys_mat_spec_dist_vs_gramian_spec_dist.png --system_matrix_type adjacency neg_laplacian --ylabel "System matrix spectral distance\nvs. Gramian spectral distance" --thresholds 0.8 0.95 --threshold_line_mode horizontal
python plot_results.py 41 44 --save_name figures/density_vs_gramian_spec_dist.png --system_matrix_type adjacency neg_laplacian --ylabel "Density versus controllability\nGramian spectral distance" --thresholds 0.8 0.95 --threshold_line_mode horizontal
python plot_results.py 25 --save_name figures/sys_mat_spec_dist_vs_density_linear_correlation.png --system_matrix_type adjacency neg_laplacian --ylabel "System matrix\nspectral distance\nversus density" --thresholds 0.8 0.95 --threshold_line_mode horizontal

