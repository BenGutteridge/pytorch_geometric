import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_stage
import torch
from .example import GNNLayer
from .utils import init_khop_nondynamic_GCN

# @register_stage('delay_gnn')      # xt+1 = f(x)       (NON-RESIDUAL)
class AlphaKGNNStage(nn.Module):
    """
    \alpha_kGNN: 
    effectively a graph rewiring, all Sk for k<r_bar considered at every t, 
    rather than being added as no of layers increase.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        num_layers (int): Number of GNN layers
    """
    def __init__(self, dim_in, dim_out, num_layers):
        super().__init__()
        # W_t only, not W_{k,t}
        # alpha_k sums to 1 and weights Sk
        # all Sk used at every layer - nondynamic
        self = init_khop_nondynamic_GCN(self, dim_in, dim_out, num_layers) # needs learned alpha_k

    def forward(self, batch):
        """
        EQUATION (2)
        \mathbf{X}^{(t + 1)} = \mathbf{X}^{(t)} + \sigma( \mathbf{S}_{k \leq \bar{r}}\mathbf{X}^{(t)}\mathbf{W}^{(t)}),
        """
        # # old k-hop method: inefficient
        # from graphgym.ben_utils import get_k_hop_adjacencies
        # k_hop_edges, _ = get_k_hop_adjacencies(batch.edge_index, self.max_k)
        # A = lambda k : k_hop_edges[k-1]

        # new k-hop method: efficient
        # k-hop adj matrix
        A = lambda k : batch.edge_index[:, batch.edge_attr==k]
        alpha = F.softmax(self.alpha, dim=0)

        # run through layers
        t = 0
        for t in range(self.num_layers):
            x = batch.x
            batch.x = torch.zeros_like(x)
            for k in range(1, self.max_k+1):
                batch.x = batch.x + alpha[k-1] * self.W[t](batch, x, A(k)).x
            batch.x = x + nn.ReLU()(batch.x)
            if cfg.gnn.l2norm: # normalises after every layer
                batch.x = F.normalize(batch.x, p=2, dim=-1)
        return batch

register_stage('alpha_k_gnn', AlphaKGNNStage)