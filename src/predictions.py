# from matplotlib import figure
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
    return model, data_list, device, target_list  # Only return the first data point
    # return model, data_list, device  # Only return the first data point


def evaluate_model(model, data, device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No gradients are needed
        data = data.to(device)
        output, attention_weights = model(data)
        edges = attention_weights[0].t().cpu().numpy()  # Transpose and convert to numpy
        scores = (attention_weights[1].squeeze().cpu().numpy())  # Squeeze and convert to numpy
        print(attention_weights[1])

    return output, edges, scores


def visualize_graph(edges, scores):
    G = nx.Graph()  
    for edge, weight in zip(edges, scores):
        G.add_edge(edge[0], edge[1], weight=weight)

    fig, ax = plt.subplots(figsize=(14, 10))


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
    return fig, ax, G, pos

def update(frame, ax, G, pos, target_list):
    ax.clear()  # Clear the previous frame
    # You can update the visualization based on the frame here
    scores = target_list[frame]  # Update with new scores or edges
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    edge_colors = plt.cm.YlOrRd(np.array(weights) / max(weights))
    
    nx.draw_networkx_nodes(G, pos, node_color="skyblue", ax=ax)
    nx.draw_networkx_edges(
        G, pos, edgelist=G.edges(), width=np.array(weights) * 5, edge_color=edge_colors, ax=ax
    )
    nx.draw_networkx_labels(G, pos, ax=ax)
    plt.title(f"Graph Attention Network Visualization - Frame {frame}")

def main():
    model_path = "checkpoints/best_model.pth"
    # data_path = "raw_data/network_results.h5"
    data_path = "raw_data/single_network_comparison.h5"
    model, data, device, target_list = load_data_and_model(model_path, data_path)
    output, edges, scores = evaluate_model(model, data[i], device)
    fig, ax, G, pos = visualize_graph(edges, scores)

    animation = FuncAnimation(
        fig = fig,
        func = update,
        frames=len(target_list),
        fargs=(ax, G, pos, target_list),
        interval=1,
        repeat=False
    )
    # plt.plot(range(0,1440,1), target_list[:1440])
    # plt.plot(target_list[1000],output.cpu().numpy())
    # plt.plot(target_list[1440], output.cpu().numpy())
    # plt.plot()
    # print(target_list)
    # plt.plot(len(output), output)
    
    # print(target_list[1000], output)
    # visualize_graph(edges, scores)
    # plt.figure(figsize=(15, 10))
    # plt.plot(target_list[1000])
    # plt.plot(output.cpu().numpy())
    # plt.xticks(range(0,33,1))
    # plt.yticks(np.arange(0.95,1,0.01))
    # plt.title('Output - Targets')


if __name__ == "__main__":
    main()
