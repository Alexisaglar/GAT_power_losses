import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as nw
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy
import h5py

# Constants
NUM_NETWORKS_TO_SIMULATE = 1  # Only simulate one network
data = 'raw_data/load_seasons.csv'

class PowerFlowSimulator:
    def __init__(self, net, load_file):
        self.net = net
        self.load_factors = self.read_load_factors(load_file)
        self.successful_nets = []
        self.all_results = {}
        self.original_loads = self.store_original_loads()

    def read_load_factors(self, file):
        data = pd.read_csv(file)
        data = data.drop(columns='datetime')
        return data

    def calculate_incidence_matrix(self):
        nodes = self.net.bus.index
        lines = self.net.line.index
        incidence_matrix = np.zeros((len(nodes), len(lines)))
        for idx, line in self.net.line.iterrows():
            incidence_matrix[line.from_bus, idx] = -1
            incidence_matrix[line.to_bus, idx] = 1
        return incidence_matrix

    def service_matrix(self):
        switch_matrix = np.zeros((len(self.net.line), len(self.net.line)))
        in_service = self.net.line['in_service']
        np.fill_diagonal(switch_matrix, in_service)
        return switch_matrix

    def is_network_radial(self, switch_matrix, incidence_matrix):
        AS_matrix = np.dot(incidence_matrix, switch_matrix)
        rank = np.linalg.matrix_rank(AS_matrix)
        if rank == len(self.net.bus.index) - 1:
            return True
        return False

    def configure_network(self):
        # Ensure all lines are in service
        # self.net.line['in_service'] = True
        return

    def run_simulation(self):
        self.configure_network()
        incidence_matrix = self.calculate_incidence_matrix()
        switch_matrix = self.service_matrix()
        if self.is_network_radial(switch_matrix, incidence_matrix):
            seasonal_results = self.simulate_loads()
            if seasonal_results:
                self.successful_nets.append(deepcopy(self.net))
                self.all_results[len(self.successful_nets)] = seasonal_results
                print(f"Successfully saved configuration {len(self.successful_nets)}.")
                self.plot_network(self.net, len(self.successful_nets))
            else:
                print("Failed to converge for all seasons.")
        else:
            print("Configured network is not radial.")

    def simulate_loads(self):
        line_data = self.net.line[['from_bus', 'to_bus', 'length_km', 'r_ohm_per_km', 'x_ohm_per_km']]
        bus_data = self.net.bus[['vn_kv', 'max_vm_pu', 'min_vm_pu']]
        seasonal_results = {'network_config': {
            'line': deepcopy(line_data[self.net.line['in_service']].values),
            'bus': deepcopy(bus_data.values.astype(float)),
        }}
        for season in self.load_factors.columns:
            time_step_results = {}
            for time_step in range(self.load_factors.shape[0]):
                self.reset_and_apply_loads(time_step, season)
                try:
                    pp.runpp(self.net, verbose=True, numba=False)
                    load_data = self.net.load[['bus','p_mw','q_mvar']]
                    slack_bus_load = pd.DataFrame([[0, 0, 0]], columns=['bus', 'p_mw', 'q_mvar'])
                    load_data = pd.concat([slack_bus_load, load_data], ignore_index=True)
                    lfa_results = {
                        'res_bus': deepcopy(self.net.res_bus.values),
                        'load': deepcopy(load_data.values.astype(float)),
                        'res_line': deepcopy(self.net.res_line[self.net.line['in_service']].values)
                    }
                    time_step_results[time_step] = lfa_results
                except pp.LoadflowNotConverged:
                    return None
            seasonal_results[season] = time_step_results
        return seasonal_results

    def store_original_loads(self):
        return self.net.load[['p_mw', 'q_mvar']].copy()

    def reset_and_apply_loads(self, time_step, season):
        self.net.load['p_mw'] = self.original_loads['p_mw']
        self.net.load['q_mvar'] = self.original_loads['q_mvar']
        scaling_factor = self.load_factors.at[time_step, season]
        self.net.load['p_mw'] *= scaling_factor
        self.net.load['q_mvar'] *= scaling_factor

    def plot_network(self, net, config_number):
        graph = pp.topology.create_nxgraph(net)
        pos = nx.spring_layout(graph, k=1, iterations=1000)
        plt.figure(figsize=(10, 6))
        nx.draw_networkx(graph, pos, with_labels=True, node_color='black', node_size=300, font_color='white')
        plt.title(f'Power Network Topology - Configuration {config_number}')
        plt.savefig(f'plots/Network_{config_number}', dpi=300)

    def save_results(self):
        with h5py.File('raw_data/33_bus_results.h5', 'w') as f:
            for net_id, net_data in self.all_results.items():
                net_group = f.create_group(f'network_{net_id}')
                static_group = net_group.create_group('network_config')
                static_group.create_dataset('line', data=net_data['network_config']['line'])
                static_group.create_dataset('bus', data=net_data['network_config']['bus'])
                for season, time_step_data in net_data.items():
                    if season == 'network_config':
                        continue
                    season_group = net_group.create_group(f'season_{season}')
                    for time_step, results in time_step_data.items():
                        time_step_group = season_group.create_group(f'time_step_{time_step}')
                        time_step_group.create_dataset('res_bus', data=results['res_bus'])
                        time_step_group.create_dataset('res_line', data=results['res_line'])
                        time_step_group.create_dataset('load', data=results['load'])

if __name__ == '__main__':
    network = nw.case33bw()
    simulator = PowerFlowSimulator(network, data)
    simulator.run_simulation()
    if simulator.successful_nets:
        simulator.save_results()

