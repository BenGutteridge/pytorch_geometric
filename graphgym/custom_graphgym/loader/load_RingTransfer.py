from torch_geometric.datasets import RingTransferDataset
from torch_geometric.graphgym.register import register_loader
from torch_geometric.graphgym.config import cfg

@register_loader('load_RingTransfer')
def load_ring_transfer_dataset(format, name, dataset_dir):
    dataset_dir = f'{dataset_dir}/{name}'
    if format == 'PyG':
        if name == 'RingTransfer':
            dataset_raw = RingTransferDataset(num_graphs=cfg.ring_dataset.num_graphs,
                                              num_nodes=cfg.ring_dataset.num_nodes,
                                              num_classes=cfg.ring_dataset.num_classes)
            return dataset_raw

