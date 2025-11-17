import sys
import pathlib
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

cfg_path = pathlib.Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
import yaml

from utils.logger import setup_logger
from data.datasets import build_dataloaders
from models.tcn import TradingModel

def get_devices() -> torch.device:

    """
    Pick CUDA if available, else MPS (Apple), else CPU.
    """

    if torch.cuda.is_available():

        return torch.device('cuda')
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():

        return torch.device('mps')
    
    return torch.device('cpu')

def set_seed(seed: int = 42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

def train_one_epoch(
        model: TradingModel,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        lambda_sigma: float
    ) -> Dict[str, float]:

    """
    Run one training epoch and return average metrics.
    """

    model.train()

    bce_loss = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_ce = 0.0
    total_vol = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:

        x = batch['x'].to(device)
        y = batch['y'].to(device)
        z = batch['z'].to(device)

        optimizer.zero_grad()

        out = model(x)
        logits = out['logits']
        sigma_hat = out['sigma']

        ce_loss = bce_loss(logits, y)

        if lambda_sigma > 0:

            v = z ** 2
            loss_vol = F.mse_loss(sigma_hat ** 2, v)
        
        else:

            loss_vol = torch.tensor(0.0, device = device)

        loss = ce_loss + (lambda_sigma * loss_vol)
        loss.backward()
        optimizer.step()

        with torch.no_grad():

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct = (preds == y).sum().item()
        
        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_ce += ce_loss.item() * batch_size
        total_vol += loss_vol.item() * batch_size
        total_correct += correct
        total_samples += batch_size

    return {
        'loss': total_loss / total_samples,
        'ce_loss': total_ce / total_samples,
        'loss_vol': total_vol / total_samples,
        'accuracy': total_correct / total_samples
    }

@torch.no_grad()
def eval_single_epoch(
    model: TradingModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    lambda_sigma: float
    ) -> Dict[str, float]:

    model.eval()

    bce_loss = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_ce = 0.0
    total_vol = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:

        x = batch['x'].to(device)
        y = batch['y'].to(device)
        z = batch['z'].to(device)

        out = model(x)
        logits = out['logits']
        sigma_hat = out['sigma']

        ce_loss = bce_loss(logits, y)

        if lambda_sigma > 0:

            v = z ** 2
            loss_vol = F.mse_loss(sigma_hat ** 2, v)
        
        else:

            loss_vol = torch.tensor(0.0, device = device)

        loss = ce_loss + (lambda_sigma * loss_vol)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        correct = (preds == y).sum().item()
        
        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_ce += ce_loss.item() * batch_size
        total_vol += loss_vol.item() * batch_size
        total_correct += correct
        total_samples += batch_size

    return {
        'loss': total_loss / total_samples,
        'ce_loss': total_ce / total_samples,
        'loss_vol': total_vol / total_samples,
        'accuracy': total_correct / total_samples
    }