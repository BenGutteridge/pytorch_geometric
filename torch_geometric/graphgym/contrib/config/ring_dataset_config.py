from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN

def ring_dataset_cfg(cfg):
    """Dataset-specific config options.
    """
    cfg.ring_dataset = CN()
    cfg.ring_dataset.num_nodes = 10
    cfg.ring_dataset.num_graphs = 6000
    cfg.ring_dataset.num_classes = 5
    cfg.ring_dataset.beta = 1 # beta, for betaGCN. Reverts to GCN for beta=1
    cfg.ring_dataset.fixed_alpha = False

register_config('ring_dataset_cfg', ring_dataset_cfg)
