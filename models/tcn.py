import sys
import pathlib
from typing import Dict
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from data.datasets import build_dataloaders

from utils.logger import setup_logger

class CausalConv1D(nn.Module):

    """
    1D causal convolution: output at time t depends only on <= t inputs.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation = 1):

        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding = self.padding,
            dilation = dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        x: [B, C_in, L] -> [B, C_out, L]
        """

        out = self.conv(x)

        if self.padding > 0:

            out = out[:, :, :-self.padding] #avoid access to the future

        return out
    
class ResidualBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 dilation: int,
                 dropout: float):
        
        super().__init__()

        self.conv1 = CausalConv1D(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1D(out_channels, out_channels, kernel_size, dilation)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size = 1)
            if in_channels != out_channels else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        x: [B, C_in, L] -> [B, C_out, L]
        """

        out = self.conv1(x)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.activation(out)

        res = x

        if self.downsample is not None:

            res = self.downsample(res)

        out = self.activation(out + res)

        return out

class TCNBackbone(nn.Module):

    """
    Stack of dilated residual causal conv blocks.
    """

    def __init__(self, 
                 in_channels: int,
                 hidden_channels: int,
                 n_blocks: int,
                 kernel_size: int,
                 dropout: float):
        
        super().__init__()

        layers = []
        channels = in_channels

        for b in range(n_blocks):

            dilation = 2 ** b

            block = ResidualBlock(
                in_channels = channels,
                out_channels = hidden_channels,
                kernel_size = kernel_size,
                dilation = dilation,
                dropout = dropout,
            )

            layers.append(block)
            channels = hidden_channels

        self.network = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        x: [B, C_in, L] -> [B, C_out, L]
        """

        return self.network(x)

class TradingModel(nn.Module):

    """
    Causal TCN + two heads:
      - classification: p_t = P(R_{t,t+H} > 0)
      - volatility:     sigma_hat_t > 0
    """

    def __init__(self, cfg: Dict):

        super().__init__()

        model_cfg = cfg["model"]
        feat_cfg = cfg['features']

        in_channels = feat_cfg['n_features']
        hidden_channels = model_cfg['hidden_channels']
        n_blocks = model_cfg['n_blocks']
        kernel_size = model_cfg['kernel_size']
        dropout = model_cfg['dropout']
        embedding_dim = model_cfg['embedding_dim']
        self.sigma_floor = model_cfg['sigma_floor']

        self.backbone = TCNBackbone(
            in_channels = in_channels,
            hidden_channels = hidden_channels,
            n_blocks = n_blocks,
            kernel_size = kernel_size,
            dropout = dropout
        )

        self.proj = nn.Linear(self.backbone.out_channels, embedding_dim)
        self.proj_act = nn.ReLU()

        self.classifier = nn.Linear(embedding_dim, 1)
        self.vol_head = nn.Linear(embedding_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        x: [B, C_in, L] (from TCNDataset)

        Returns:
            embedding: [B, q]
            logits:    [B]
            p:         [B]
            sigma:     [B]
        """

        h = self.backbone(x)

        h_last = h[:, :, -1]

        e = self.proj_act(self.proj(h_last))

        logits = self.classifier(e).squeeze(-1)

        p = torch.sigmoid(logits)

        u = self.vol_head(e).squeeze(-1)

        sigma_hat = F.softplus(u) + self.sigma_floor

        return {
            'embedding': e,
            'logits': logits,
            'p': p,
            'sigma': sigma_hat
        }