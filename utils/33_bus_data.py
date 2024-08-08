import h5py
import numpy as np
import pandapower as pp
import pandapower.networks as pn
import torch
from torch_geometric.data import Data


def run_power_flow():
    net = pn.case33bw()
    pp.runpp(net)
    return net


def extract_data_from_pandapower(net):
    # Extract line data (from_bus, to_bus, x, r, length)
    line_data = np.array(
        [
            net.line.from_bus.values,
            net.line.to_bus.values,
            net.line.x_ohm_per_km.values,
            net.line.r_ohm_per_km.values,
            net.line.length_km.values,
        ]
    ).T

    line_res_data = np.array(
        [
            net.res_line.p_from_mw.values,
            net.res_line.q_from_mvar.values,
            net.res_line.pl_mw.values,
            net.res_line.ql_mvar.values,
        ]
    ).T

    # Extract bus data (voltage magnitude, angle)
    bus_data = np.array([net.res_bus.vm_pu.values, net.res_bus.va_degree.values]).T

    # Extract load data (p_mw, q_mvar)
    # Generate a 33x2 array, to account for slack bus values
    # load_data = np.zeros((len(net.bus), 2))
    # load_data[net.load.bus.values] = np.array([net.load.p_mw.values, net.load.q_mvar.values]).T
    load_data = np.array([net.res_bus.p_mw.values, net.res_bus.q_mvar.values]).T
    print(load_data)

    # Extract only data in_service
    bus_data = bus_data[net.bus["in_service"].values]
    line_data = line_data[net.line["in_service"].values]
    load_data = load_data[net.bus["in_service"].values]
    line_res_data = line_res_data[net.res_line["in_service"].values]

    return line_data, bus_data, load_data, line_res_data


def save_data_to_h5py(file, line_data, bus_data, load_data):
    with h5py.File(file, "w") as f:
        net_group = f.create_group("network_1")
        network_config = net_group.create_group("network_config")

        # Save static data
        network_config.create_dataset("line", data=line_data)
        network_config.create_dataset("bus", data=bus_data)

        # Save dynamic data
        season_group = net_group.create_group("season_winter")
        time_step_group = season_group.create_group("time_step_1")
        time_step_group.create_dataset("load", data=load_data)
        time_step_group.create_dataset("res_bus", data=bus_data)


def load_network_data(file, network_key):
    with h5py.File(file, "r") as f:
        net_group = f[network_key]
        static_data = {
            "line": net_group["network_config/line"][:],
            "bus": net_group["network_config/bus"][:],
        }
    return static_data


def create_dataset_from_h5py(file):
    data_list = []
    target_list = []

    with h5py.File(file, "r") as f:
        network_keys = [key for key in f.keys() if key.startswith("network_")]

        for network_key in network_keys:
            static_data = load_network_data(file, network_key)
            net_group = f[network_key]
            season_keys = ["season_winter"]
            for season_key in season_keys:
                if season_key in net_group:
                    season_group = net_group[season_key]
                    for time_step_key in season_group.keys():
                        time_step_group = season_group[time_step_key]

                        # extract edge features (x, r, length)
                        line_data = static_data["line"]
                        edge_features = line_data[:, 2:5]

                        # extract target values (voltage magnitude and angle)
                        target_bus = time_step_group["res_bus"][:, :2]

                        # extract node features (p and q)
                        node_features = time_step_group["load"][:, :2]

                        # create edge index  (from_bus and to_bus in first two columns)
                        edge_index = np.vstack(
                            (line_data[:, 0], line_data[:, 1])
                        ).astype(int)

                        # convert to torch tensors
                        node_features = torch.tensor(node_features, dtype=torch.float)
                        edge_features = torch.tensor(edge_features, dtype=torch.float)
                        edge_index = torch.tensor(edge_index, dtype=torch.long)
                        targets = torch.tensor(target_bus, dtype=torch.float)

                        # create data object
                        data = Data(
                            x=node_features,
                            edge_index=edge_index,
                            edge_attr=edge_features,
                        )
                        data_list.append(data)
                        # target_list.append(targets)
                        target_list.append(target_bus)

    return data_list, target_list


# Run the power flow and extract data
net = run_power_flow()
line_data, bus_data, load_data, line_res_data = extract_data_from_pandapower(net)

print(line_data, bus_data, load_data)
# Save the data to an HDF5 file
h5_file = "power_flow_data.h5"
save_data_to_h5py(h5_file, line_data, bus_data, load_data)

# Create dataset from the HDF5 file
data_list, target_list = create_dataset_from_h5py(h5_file)
print(data_list, target_list)
