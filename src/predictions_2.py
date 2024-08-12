import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
import numpy as np
import torch
from data_loader import create_dataset
from model import GATNet
from torch_geometric.data import Data

def load_data_and_model(model_path, data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GATNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    data_list, target_list = create_dataset(data_path)
    return model, data_list, device, target_list  # Return the full dataset


def evaluate_model(model, data, device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No gradients are needed
        data = data.to(device)
        output, attention_weights = model(data)
        edges = attention_weights[0].t().cpu().numpy()  # Transpose and convert to numpy
        scores = (attention_weights[1].squeeze().cpu().numpy())  # Squeeze and convert to numpy

    return output, edges, scores


def visualize_graph(edges, scores, ax, sm, pos, frame):
    G = nx.Graph()
    for edge, weight in zip(edges, scores):
        G.add_edge(edge[0], edge[1], weight=weight)

    ax.clear()  # Clear the axes before plotting

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    edge_colors = plt.cm.YlOrRd(np.array(weights) / max(weights))  # Normalize for coloring

    nx.draw_networkx_nodes(G, pos, node_color="skyblue", ax=ax)
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edges,
        width=np.array(weights) * 5,
        edge_color=edge_colors,
        ax=ax
    )
    nx.draw_networkx_labels(G, pos, ax=ax)

    # Enhanced colorbar
    # sm = plt.cm.ScalarMappable(
    #     cmap=plt.cm.YlOrRd, norm=plt.Normalize(vmin=min(scores), vmax=max(scores))
    # )
    # sm.set_array([])
    # if ax.collections:  # Check if there is a colorbar already and remove it
    #     ax.collections.clear()
    # plt.colorbar(sm, orientation="vertical", label="Attention Weights", ax=ax)

    # Display frame number
    # ax.text(0.05, 0.95, f'Frame: {frame}', transform=ax.transAxes, fontsize=12, verticalalignment='top')

    ax.set_title(f"Graph Attention Network Visualization{frame}")


def update(frame, model, data_list, device, ax, sm, pos):
    output, edges, scores = evaluate_model(model, data_list[frame], device)
    visualize_graph(edges, scores, ax, sm, pos, frame)
    return ax


def main():
    model_path = "checkpoints/best_model.pth"
    data_path = "raw_data/single_network_comparison.h5"
    model, data_list, device, target_list = load_data_and_model(model_path, data_path)

    # Precompute the layout for the graph to fix node positions
    first_data = data_list[0].to(device)
    _, edges, scores = evaluate_model(model, first_data, device)
    G = nx.Graph()
    for edge in edges:
        G.add_edge(edge[0], edge[1])
    pos = nx.spring_layout(G, k=0.9, iterations=600)  # Compute layout only once

    fig, ax = plt.subplots(figsize=(14, 10))
    sm = None
    
    animation = FuncAnimation(
        fig,
        update,
        fargs=(model, data_list, device, ax, sm, pos),
        frames=len(data_list),  # Iterate through all data points
        interval=0.1,  # Adjust the interval as needed
        repeat=False
    )

    plt.show()


if __name__ == "__main__":
    main()

