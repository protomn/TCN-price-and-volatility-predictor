import sys
import pathlib
from typing import Dict, Tuple
import yaml

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from utils.logger import setup_logger
from data.loader import DataLoader as PriceDataLoader
from data.features import make_datasets

class TCNDataset(Dataset):

    """
    Wraps (X, y, z) numpy arrays into a PyTorch Dataset.

    X: [N, lookback, n_features]  (from make_datasets)
    y: [N,]  (0/1 labels)
    z: [N,]  (horizon log-returns)
    """

    def __init__(self, X: np.ndarray,
                 y: np.ndarray,
                 z: np.ndarray):
        
        assert X.ndim == 3, f"Expected X to be 3D, got shape {X.shape}"

        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.z = z.astype(np.float32)

        self.n_samples, self.lookback, self.n_features = self.X.shape

    def __len__(self) -> int:

        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        """
        Return a single sample:

        - x: [n_features, lookback]  (channels-first for Conv1d)
        - y: scalar label
        - z: scalar horizon log-return
        """

        x_np = self.X[idx]
        x_np = x_np.T

        x = torch.from_numpy(x_np)
        y = torch.tensor(self.y[idx], dtype = torch.float32)
        z = torch.tensor(self.z[idx], dtype = torch.float32)

        return {'x': x, 'y': y, 'z': z}
    
def build_dataloaders(cfg: Dict) -> Tuple[DataLoader,
                                              DataLoader,
                                              DataLoader]:
        
    """
    High-level helper to go from config -> DataLoaders.

    Returns:
    train_loader, val_loader, test_loader
    """

    logger = setup_logger(__name__, cfg.get("logging", {}).get("level", "INFO"))

    logger.info("Fetching raw price data for dataloaders...")
    price_loader = PriceDataLoader(cfg)
    price_df = price_loader.fetch_data()

    train_arr, val_arr, test_arr = make_datasets(price_df, cfg)

    train_ds = TCNDataset(train_arr['X'], train_arr['y'], train_arr['z'])
    val_ds = TCNDataset(val_arr['X'], val_arr['y'], val_arr['z'])
    test_ds = TCNDataset(test_arr['X'], test_arr['y'], test_arr['z'])

    logger.info(
        f"Train dataset: {len(train_ds)} samples, "
        f"Val: {len(val_ds)}, Test: {len(test_ds)}"
    )

    batch_size = 64

    train_loader = DataLoader(
                train_ds,
                batch_size = batch_size,
                shuffle = True,
                num_workers = 0,
                drop_last = False,
                )
        
    val_loader = DataLoader(
                val_ds,
                batch_size = batch_size,
                shuffle = False,
                num_workers = 0,
                drop_last = False,
                )
        
    test_loader = DataLoader(
                test_ds,
                batch_size = batch_size,
                shuffle = False,
                num_workers = 0,
                drop_last = False,
                )
        
    return train_loader, val_loader, test_loader
    
if __name__ == "__main__":

    cfg_path = pathlib.Path(__file__).resolve().parents[1] / "configs" / "default.yaml"

    with open(cfg_path, "r") as f:

        cfg = yaml.safe_load(f)

    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    print("Number of train batches:", len(train_loader))
    
    batch = next(iter(train_loader))
    x, y, z = batch["x"], batch["y"], batch["z"]

    print("Batch x shape:", x.shape)  
    print("Batch y shape:", y.shape)  
    print("Batch z shape:", z.shape)  