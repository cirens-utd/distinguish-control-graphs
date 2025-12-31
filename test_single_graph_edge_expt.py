from modules.single_graph_edge_expt import single_graph_edge_expt

options =  {'graph_choice': {'type': 'connected_ER', 'n': 20, 'p': 0.5},
            'graph_matrix_choice': 'laplacian',
            'edge_score_choice': 'sys_mat_spec_dist', 
            'input': 'zfs',
            't_horizon': 1, 
            'plots': [{'y1': 'sys_mat_spec_dist', 'y2': 'Wc_spec_dist'}]}
single_graph_edge_expt(options)