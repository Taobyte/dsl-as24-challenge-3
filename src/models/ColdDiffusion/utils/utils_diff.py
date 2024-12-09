from torch.utils.data import DataLoader

from models.ColdDiffusion.dataset import CDiffDataset

def create_dataloader(cfg, is_noise=False):
    
    # Step 1: Load the data
    if is_noise:
        train_dataset = CDiffDataset(cfg.user.data.datapath + "train_noise_003.npy")
        val_dataset = CDiffDataset(cfg.user.data.datapath + "val_noise_003.npy")
    else:
        train_dataset = CDiffDataset(cfg.user.data.datapath + "train_eq_003.npy")
        val_dataset = CDiffDataset(cfg.user.data.datapath + "val_eq_003.npy")
    
    # Step 2: Creating DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=cfg.model.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg.model.batch_size, shuffle=False)
    
    # test_dataset = None
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader #, test_loader, index_to_trace_name