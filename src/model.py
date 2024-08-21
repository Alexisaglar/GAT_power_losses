import torch
import torch.nn.functional as F
# from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import RGCNConv 
from torch_geometric.nn import GraphSAGE 

# class GATNet(torch.nn.Module):
#     def __init__(self):
#         super(GATNet, self).__init__()
#         self.conv_layer1 = GATv2Conv(
#             in_channels=2, out_channels=16, heads=8, concat=True, edge_dim=7
#         )
#         self.conv_layer2 = GATv2Conv(
#             in_channels=16 * 8, out_channels=64, heads=8, concat=True, edge_dim=7
#         )
#         self.conv_layer3 = GATv2Conv(
#             in_channels=64 * 8, out_channels=128, heads=8, concat=True, edge_dim=7
#         )
#         self.conv_layer4 = GATv2Conv(
#             in_channels=128 * 8, out_channels=64, heads=4, concat=True, edge_dim=7
#         )
#         self.conv_layer5 = GATv2Conv(
#             in_channels=64 * 4, out_channels=32, heads=2, concat=True, edge_dim=7
#         )
#         self.conv_layer6 = GATv2Conv(
#             in_channels=32 * 2, out_channels=1, heads=1, concat=False, edge_dim=7
#         )
#
#     def forward(self, data):
#         X, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
#
#         # Process each layer sequentially
#         X = self.conv_layer1(X, edge_index, edge_attr)
#         X = F.elu(X)
#
#         X = self.conv_layer2(X, edge_index, edge_attr)
#         X = F.elu(X)
#
#         X = self.conv_layer3(X, edge_index, edge_attr)
#         X = F.elu(X)
#
#         X = self.conv_layer4(X, edge_index, edge_attr)
#         X = F.elu(X)
#
#         X = self.conv_layer5(X, edge_index, edge_attr)
#         X = F.elu(X)
#
#         X, attention_weights = self.conv_layer6(X, edge_index, edge_attr, return_attention_weights=True)
#
#         return X, attention_weights
#
#

class GraphSAGENet(torch.nn.Module):
    def __init__(self):
        super(GraphSAGENet, self).__init__()
        self.sage = GraphSAGE(
            in_channels=2,             # Number of input features
            hidden_channels=256,        # Size of hidden layers
            num_layers=12,               # Number of layers in the model
            out_channels=1,             # Size of the output layer
            dropout=0.0,                # Dropout probability (set to 0.0 for no dropout)
        )

    def forward(self, data):
        X, edge_index = data.x, data.edge_index

        # Perform the GraphSAGE computation
        X = self.sage(X, edge_index)

        return X
