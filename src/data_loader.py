import numpy as np
import torch
from torch_geometric.data import Data
import h5py

def load_network_data(file, network_key):
    with h5py.File(file, 'r') as f:
        net_group = f[network_key]
        static_data = {
            'line': net_group['network_config/line'][:],
            'bus': net_group['network_config/bus'][:]
        }
    return static_data

def create_dataset(file):
    data_list = []
    target_list = []

    with h5py.File(file, 'r') as f:
        network_keys = [key for key in f.keys() if key.startswith('network_')]
        
        for network_key in network_keys:
            static_data = load_network_data(file, network_key)
            net_group = f[network_key]
            season_keys = ['season_winter', 'season_spring', 'season_summer', 'season_autumn']
            for season_key in season_keys:
                if season_key in net_group:
                    season_group = net_group[season_key]
                    for time_step_key in season_group.keys():
                        time_step_group = season_group[time_step_key]

                        ## extract edge features (x, r, length)
                        line_network_data = static_data['line']
                        line_network_data = line_network_data[:, 2:5] 
                        line_features = time_step_group['res_line'][:, [0,1,4,5]]
                        edge_features = np.concatenate((line_network_data, line_features), axis=1)

                        ## create edge index  (from_bus and to_bus in first two columns)
                        edge_index = np.vstack((static_data['line'][:, 0], static_data['line'][:, 1])).astype(int)

                        ## extract target values (voltage magnitude and angle)
                        # target_bus = time_step_group['res_line'][:, [4,5]]  
                        # target_line_losses = time_step_group['res_line'][:, [4,5]]  
                        target_bus = time_step_group['res_bus'][:, [0]]
                        
                        
                        # extract node features (p and q)
                        node_features = time_step_group['res_bus'][:, [2,3]]
                        # node_features = np.concatenate((line_features, bus_features))


                        # convert to torch tensors
                        node_features = torch.tensor(node_features, dtype=torch.float)
                        edge_features = torch.tensor(edge_features, dtype=torch.float)
                        edge_index = torch.tensor(edge_index, dtype=torch.long)
                        targets = torch.tensor(target_bus, dtype=torch.float)
                        
                        # create data object
                        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)
                        data_list.append(data)
                        target_list.append(targets)

    return data_list, target_list

# create_dataset('raw_data/33_bus_with_pl_ql.h5')
