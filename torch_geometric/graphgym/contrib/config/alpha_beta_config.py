from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


def set_cfg_alpha_beta(cfg):
    r'''
    This function sets the default config value for customized options
    :return: customized configuration use by the experiment.
    '''

    # ----------------------------------------------------------------------- #
    # Customized options
    # ----------------------------------------------------------------------- #

    # max rbar for the betaGCN model - i.e. vanilla GCN w/ adjacency for FC graph up to beta hops
    cfg.beta = 1
    # max no. of hops to aggregate over in every layer for alphaGCN
    cfg.alpha = 1 # defaults to n_layers if this is not changed

    # # if True: for alphaGCN, rather than having alpha as a learnable vector summing to 1, just use a 1-vector
    # cfg.fixed_alpha = False

    # # if True: alphaGCN uses W_{k,t} like r*GCN rather than just W_t, and alpha is alpha_{k,t}
    # cfg.alpha_W_kt = False


register_config('alpha_beta', set_cfg_alpha_beta)
