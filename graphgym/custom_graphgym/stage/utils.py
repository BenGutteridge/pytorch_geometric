import torch.nn as nn
# import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
# from torch_geometric.graphgym.register import register_stage
import torch
from .example import GNNLayer


def init_khop_GCN(model, dim_in, dim_out, num_layers):
  """The k-hop GCN param initialiser, used for k_gnn and delay_gnn"""
  model.num_layers = num_layers
  model.max_k = cfg.gnn.layers_mp # cfg.delay.max_k
  for t in range(num_layers):
      d_in = dim_in if t == 0 else dim_out
      K = min(model.max_k, t+1)
      for k in range(1, K+1):
          W = GNNLayer(d_in, dim_out) # regular GCN layers
          model.add_module('W_k{}_t{}'.format(k,t), W)
  return model


def init_khop_LiteGCN(model, dim_in, dim_out, num_layers):
  """The lightweight version of the k-hop GCN, with nu_{k,t}W_k instead of W_{k,t}, and W instead of W(t).
  Will be used for delite_gnn and klite_gnn"""
  model.num_layers = num_layers
  model.max_k = cfg.gnn.layers_mp # cfg.delay.max_k
  # make the W_k
  W = []
  for k in range(model.max_k):
      W.append(GNNLayer(dim_in, dim_out))
  model.W = nn.ModuleList(W)
  # make K*T nu_{k,t} scalars
  nu = {}
  for t in range(num_layers):
    for k in range(1, min(model.max_k, t+1)+1):
      nu['%d,%d'%(k,t)] = nn.parameter.Parameter(torch.Tensor(1))
  model.nu = nn.ParameterDict(nu)

  return model


def init_khop_nondynamic_GCN(model, dim_in, dim_out, num_layers, fixed_alpha=False):
  """For the non-dynamic k-hop model: alpha_k_gnn.
  W_t, alpha_k (sums to 1)"""
  model.num_layers = num_layers
  model.max_k = cfg.gnn.layers_mp # cfg.delay.max_k
  # make the W_k
  W = []
  for t in range(num_layers):
      W.append(GNNLayer(dim_in, dim_out))
  model.W = nn.ModuleList(W)
  if fixed_alpha:
    print('Using fixed alpha model.')
    model.alpha = torch.ones(model.max_k) # 1-vector
  else:
    print('Using learnable sum-to-1 alpha model.')
    model.alpha = nn.parameter.Parameter(torch.ones(model.max_k)/model.max_k)
  return model