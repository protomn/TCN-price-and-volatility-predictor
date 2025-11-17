import numpy as np
import pandas as pd
import sys, pathlib
from typing import Dict
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from utils.logger import setup_logger

def _col_to_series(df: pd.DataFrame, col_name: str) -> pd.Series:
    """
    Ensure df[col_name] is a 1D Series.

    yfinance sometimes returns a DataFrame with shape (n, 1) if you have
    multi-level columns (e.g. multiple tickers). For now we support only
    a single asset and collapse that to a Series.
    """
    col = df[col_name]
  
    if isinstance(col, pd.DataFrame):

        if col.shape[1] != 1:

            raise ValueError(
                f"{col_name} has {col.shape[1]} columns; "
                "compute_raw_features currently supports a single asset."
            )
        
        col = col.iloc[:, 0]
    return col

def compute_raw_features(price_df: pd.DataFrame) -> pd.DataFrame:

    """
    Take OHLCV data and compute 12 causal technical features per bar.

    price_df: DataFrame with at least columns
              ['Open', 'High', 'Low', 'Close', 'Volume'] and a DateTimeIndex.

    Returns: DataFrame with the same index and 12 feature columns.
    """

    df = price_df.copy()

    close = _col_to_series(df, "Close")
    high = _col_to_series(df, "High")
    low = _col_to_series(df, "Low")
    volume = _col_to_series(df, "Volume")

    df['log_ret'] = np.log(close / close.shift(1))
    delta = close.diff()

    #1. 1-bar Log Return

    f1 = df['log_ret']

    #2. 5-bar Log momentum

    f2 = df['log_ret'].rolling(window = 5).sum()

    #3. 20-bar Log Momentum

    f3 = df['log_ret'].rolling(window = 20).sum()

    #4. Price v. 10-bar SMA

    sma_10 = close.rolling(window = 10).mean()
    f4 = (close - sma_10) / sma_10

    #5. Price v. 20-bar SMA

    sma_20 = close.rolling(window = 20).mean()
    f5 = (close - sma_20) / sma_20

    #6. 5-bar Realized Volatility

    f6 = np.sqrt(df['log_ret'].pow(2).rolling(window = 5).mean())

    #7. 20-bar realized volatility

    f7 = np.sqrt(df['log_ret'].pow(2).rolling(window = 20).mean())

    #8. Intraday range / close

    f8 = (high - low) / close

    #9. Volume z-score (20-bars)

    vol_ma20 = volume.rolling(window = 20).mean()
    vol_std20 = volume.rolling(window = 20).std(ddof = 0)
    f9 = (volume - vol_ma20) / vol_std20

    #10. Volume z-score (60 bars)

    vol_ma60 = volume.rolling(window = 60).mean()
    vol_std60 = volume.rolling(window = 60).std(ddof = 0)
    f10 = (volume - vol_ma60) / vol_std60

    #11. 14-bar RSI

    gains = delta.clip(lower = 0)
    losses = -delta.clip(upper = 0)

    av_gains = gains.ewm(alpha = 1/14, adjust = False).mean()
    av_losses = losses.ewm(alpha = 1/14, adjust = False).mean()

    rs = av_gains / av_losses.replace(0, np.nan)
    rsi = 100 - (100/(1+rs))
    f11 = (rsi - 50) / 50

    #12. MACD momentum
    
    ema_fast = close.ewm(span = 12, adjust = False).mean()
    ema_slow = close.ewm(span = 26, adjust = False).mean()

    macd = ema_fast - ema_slow
    f12 = macd / ema_slow.replace(0, np.nan)

    features = pd.DataFrame(
        {
            'f1_ret_1b': f1,
            'f2_ret_5b': f2,
            'f3_mom_20b': f3,
            'f4_px_sma10': f4,
            'f5_px_sma20': f5,
            'f6_rv_5b': f6,
            'f7_rv_20b': f7,
            'f8_range': f8,
            'f9_vol_z20': f9,
            'f10_vol_z60': f10,
            'f11_rsi_14b': f11,
            'f12_macd_mom': f12
        },
        index = df.index
    )

    return features


def normalize_features(features: pd.DataFrame,
                       norm_window: int) -> pd.DataFrame:
    
    """
    Rolling z-score normalization over 'norm_window' bars.

    We compute mean and std over the past 'norm_window' observations
    (including t), then (f - mean)/std.
    """

    rolling_mean = features.rolling(window = norm_window).mean()
    rolling_std = features.rolling(window = norm_window).std()

    rolling_std = rolling_std.replace(0, np.nan)

    norm = (features - rolling_mean) / rolling_std

    norm = norm.dropna()

    return norm


def build_targets(price_df: pd.DataFrame,
                  horizon: int) -> pd.DataFrame:
    
    df = price_df.copy()
    close = _col_to_series(df, "Close")

    log_ret = np.log(close / close.shift(1))

    Z = log_ret.rolling(window = horizon).sum().shift(-horizon)
    Y = (Z > 0).astype(int)

    targets = pd.DataFrame({"Z": Z, "Y": Y}, index = df.index)

    return targets

def make_datasets(price_df: pd.DataFrame,
                 cfg: Dict):
    
    """
    Main entry point:

    - price_df: OHLCV data (from DataLoader)
    - cfg: full config dict (from default.yaml)

    Returns:
        train, val, test: each a dict with
            'X': np.ndarray [n_samples, lookback, n_features]
            'y': np.ndarray [n_samples,]
            'z': np.ndarray [n_samples,]
            'dates': np.ndarray of timestamps (decision times)
    """

    logger = setup_logger(__name__, cfg.get("logging", {}).get("level", "INFO"))

    data_cfg = cfg['data']
    feat_cfg = cfg['features']

    lookback = cfg['data']['lookback_window']
    horizon = cfg['data']['horizon']
    norm_window = cfg['features']['normalization_window']

    #1. Compute features
    logger.info("Computing raw features:")
    raw_feats = compute_raw_features(price_df)

    #2. Normalize features
    logger.info("Normalizing features:")
    norm_feats = normalize_features(raw_feats, norm_window)

    #3. Build targets
    logger.info("Building targets:")
    targets = build_targets(price_df, horizon)

    #4. Align and Clean features and targets
    logger.info("Aligning and cleaning features and targets")
    combined = norm_feats.join(targets, how = "inner")

    combined = combined.dropna(subset = ["Z", "Y"])

    feature_cols = [c for c in combined.columns if c.startswith("f")]

    assert len(feature_cols) == feat_cfg["n_features"], \
    f"Expected {feat_cfg['n_features']}, got {len(feature_cols)}"

    #5. Build windows
    logger.info("Building windows")
    X_list = []
    y_list = []
    z_list = []
    date_list = []

    values = combined[feature_cols].values

    Z_values = combined["Z"].values
    Y_values = combined['Y'].values

    index = combined.index

    for t in range(lookback - 1, len(combined)):

        window = values[t - lookback + 1 : t + 1, :]
        X_list.append(window)
        y_list.append(Y_values[t])
        z_list.append(Z_values[t])
        date_list.append(index[t])

    X = np.stack(X_list, axis = 0)
    y = np.array(y_list)
    z = np.array(z_list)
    dates = np.array(date_list)

    logger.info(f"Total samples after windowing: {X.shape[0]}")

    #6. Time based train/val/test split

    train_end = pd.to_datetime(data_cfg["train_end"])
    val_end = pd.to_datetime(data_cfg['val_end'])

    date_series = pd.to_datetime(dates)

    train_mask = date_series <= train_end
    val_mask = (date_series > train_end) & (date_series <= val_end)
    test_mask = date_series > val_end

    def subset(mask):

        return {
            'X': X[mask],
            'y': y[mask],
            'z': z[mask],
            'dates': dates[mask]
        }
    
    train = subset(train_mask)
    val = subset(val_mask)
    test = subset(test_mask)

    logger.info(
        f"Train samples: {train['X'].shape[0]}, "
        f"Val: {val['X'].shape[0]}, Test: {test['X'].shape[0]}"
    )

    return train, val, test