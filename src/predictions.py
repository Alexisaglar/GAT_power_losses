from matplotlib import figure
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
import numpy as np
import torch
from data_loader import create_dataset
from model import GraphSAGENet 
from torch_geometric.data import Data

def load_data_and_model(model_path, data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphSAGENet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    data_list, target_list = create_dataset(data_path)
    return model, data_list[1000], device, target_list  # Only return the first data point


def evaluate_model(model, data, device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No gradients are needed
        data = data.to(device)
        output = model(data)
        # output, attention_weights = model(data)
        # edges = attention_weights[0].t().cpu().numpy()  # Transpose and convert to numpy
        # scores = (attention_weights[1].squeeze().cpu().numpy())  # Squeeze and convert to numpy
        # print(attention_weights[1])

    # return output, edges, scores
    return output


def visualize_graph(edges, scores):
    G = nx.Graph()
    for edge, weight in zip(edges, scores):
        G.add_edge(edge[0], edge[1], weight=weight)

    plt.figure(figsize=(14, 10))
    ax = plt.gca()

    pos = nx.spring_layout(G, k=0.9, iterations=600, weight="weight")
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    edge_colors = plt.cm.YlOrRd(np.array(weights) / max(weights))  # Normalize for coloring

    nx.draw_networkx_nodes(G, pos, node_color="skyblue")

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edges,
        width=np.array(weights) * 5,
        edge_color=edge_colors,
    )

    nx.draw_networkx_labels(G, pos)

    # Enhanced colorbar
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.YlOrRd, norm=plt.Normalize(vmin=min(scores), vmax=max(scores))
    )
    sm.set_array([])
    plt.colorbar(sm, orientation="vertical", label="Attention Weights", ax=ax)
    plt.title("Graph Attention Network Visualization")
    plt.savefig(f'plots/attention_visualisation', dpi=300)
    # plt.show()
    return plt


def main():
    model_path = "checkpoints/best_model.pth"
    # data_path = "raw_data/network_results.h5"
    data_path = "raw_data/single_network_comparison.h5"
    # data_path = "raw_data/random_loads_single_network.h5"
    model, first_data, device, target_list = load_data_and_model(model_path, data_path)
    # output, edges, scores = evaluate_model(model, first_data, device)
    output = evaluate_model(model, first_data, device)


    # Ensure both output and target_list[1000] are on the CPU and converted to NumPy arrays
    if torch.is_tensor(output):
        output = output.cpu().numpy()

    target = target_list[1000].cpu().numpy() if torch.is_tensor(target_list[1000]) else target_list[1000]

    # Plotting the data
    plt.plot(output)
    plt.plot(target)
    plt.xticks(np.arange(0,len(output), step=1))
    plt.yticks(np.arange(0.9,1.15, step=0.05))
    plt.show()


    # visualize_graph(edges, scores)
    # print(scores)
    plt.show()


if __name__ == "__main__":
    main()
