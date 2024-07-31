import matplotlib.pyplot as plt
import networkx as nx
import pandapower as pp
import pandapower.networks as nw
import seaborn as sns


def power_flow(network):
    results = pp.runpp(network, verbose=True, numba=False)
    return results

def plot_network(network):
    graph = pp.topology.create_nxgraph(network)
    pos = nx.spring_layout(graph, k=1, iterations=1000)
    plt.figure(figsize=(10, 6))
    nx.draw_networkx(graph, pos, with_labels=True, node_color='black', node_size=300, font_color='white')
    plt.show()

def normalised_plot(data_frame, name):
    normalized_df = data_frame.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)

    # Create the heatmap
    plt.figure(figsize=(15, 10))
    sns.heatmap(normalized_df, cmap='viridis', annot=False)  # Use annot=True if you want to see the values
    plt.title('Column-Normalized Heatmap')
    plt.show()
    plt.savefig(f'plots/results_{name}', dpi=300)

if __name__ == '__main__':
    # results = power_flow(nw.case33bw)
    network = nw.case33bw()
    pp.runpp(network, algorithm="bfsw", numba=False)
    print(network.res_line, network.res_bus)
    plot_network(network)
    normalised_plot(network.res_line, "lines")
    normalised_plot(network.res_bus, "bus")
    normalised_plot(network.res_load, "load")
