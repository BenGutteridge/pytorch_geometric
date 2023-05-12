from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


def set_cfg_edge_labels(cfg):
    r'''
    This function sets the default config value for customized options
    :return: customized configuration use by the experiment.
    '''

    # ----------------------------------------------------------------------- #
    # Customized options
    # ----------------------------------------------------------------------- #

    # max rbar for the betaGCN model - i.e. vanilla GCN w/ adjacency for FC graph up to beta hops
    cfg.use_edge_labels = False
    cfg.edge_types = [] # fills in automatically


register_config('edge_labels', set_cfg_edge_labels)
