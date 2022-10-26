import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym import register
from torch_geometric.graphgym.register import register_stage
# UPDATED FROM gnn.py
# from torch_geometric.graphgym.models.gnn import GNNLayer 
from .example import GNNLayer
from .utils import init_khop_GCN

@register_stage('delay_gnn')      # xt+1 = f(x)       (NON-RESIDUAL)
class DelayGNNStage(nn.Module):
    """
    Stage that stack GNN layers and includes a 1-hop skip (Delay GNN for max K = 2)

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
        x_{t+1} = x_t + f(x_t, x_{t-1})
        first pass: uses regular edge index for each layer
        """
        # # old k-hop method: inefficient
        # from ben_utils import get_k_hop_adjacencies
        # k_hop_edges, _ = get_k_hop_adjacencies(batch.edge_index, self.max_k)
        # A = lambda k : k_hop_edges[k-1]

        # new k-hop method: efficient
        # k-hop adj matrix
        A = lambda k : batch.edge_index[:, batch.edge_attr==k]
        
        # run through layers
        t, x = 0, [] # length t list with x_0, x_1, ..., x_t
        modules = self.children()
        for t in range(self.num_layers):
            x.append(batch.x)
            batch.x = torch.zeros_like(x[t])
            for k in range(1, (t+1)+1):
                W = next(modules)
                batch.x = batch.x + W(batch, x[t+1-k], A(k)).x
            batch.x = x[t] + nn.ReLU()(batch.x)
            if cfg.gnn.l2norm: # normalises after every layer
                batch.x = F.normalize(batch.x, p=2, dim=-1)
        return batch
