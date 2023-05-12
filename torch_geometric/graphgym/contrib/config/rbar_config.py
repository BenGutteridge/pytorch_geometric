from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


def set_cfg_rbar(cfg):
    r'''
    This function sets the default config value for customized options
    :return: customized configuration use by the experiment.
    '''

    # ----------------------------------------------------------------------- #
    # Customized options
    # ----------------------------------------------------------------------- #

    # example argument
    cfg.rbar = 1
    cfg.rbar_v2 = False # determines whether to use (k-1)//rbar as delay (True), or max(k, 0) (False)


register_config('rbar', set_cfg_rbar)
