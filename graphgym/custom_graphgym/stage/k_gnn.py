import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_stage
import torch
from .example import GNNLayer
from .utils import init_khop_GCN

# @register_stage('k_gnn')      # to compare with DelayGCN: all x is x(t), no 'delay', same params
class K_GNNStage(nn.Module):
    """
    NO DELAY ELEMENT

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        num_layers (int): Number of GNN layers
    """
    def __init__(self, dim_in, dim_out, num_layers):
        super().__init__()
        self = init_khop_GCN(self, dim_in, dim_out, num_layers)

    def forward(self, batch):
        """
        x_{t+1} = x_t + f(x_t)
        first pass: uses regular edge index for each layer
        """
        # # old k-hop method: inefficient
        # from graphgym.ben_utils import get_k_hop_adjacencies
        # k_hop_edges, _ = get_k_hop_adjacencies(batch.edge_index, self.max_k)
        # A = lambda k : k_hop_edges[k-1]

        # new k-hop method: efficient
        # k-hop adj matrix
        A = lambda k : batch.edge_index[:, batch.edge_attr==k]
        
        # run through layers
        t = 0
        modules = self.children()
        for t in range(self.num_layers):
            x = batch.x
            batch.x = torch.zeros_like(x)
            for k in range(1, (t+1)+1):
                W = next(modules)
                batch.x = batch.x + W(batch, x, A(k)).x
            batch.x = x + nn.ReLU()(batch.x)
            if cfg.gnn.l2norm: # normalises after every layer
                batch.x = F.normalize(batch.x, p=2, dim=-1)
        return batch

register_stage('k_gnn', K_GNNStage)
