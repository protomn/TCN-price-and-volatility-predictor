import yfinance as yf
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import yaml

import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from utils.logger import setup_logger

class DataLoader:

    """
    Responsible for handling data fetching.
    """

    def __init__(self, config: dict):

        self.config = config['data']
        self.logger = setup_logger(__name__)

    def fetch_data(self, symbol: Optional[str] = None) -> pd.DataFrame:

        """
        Fetch OHLCV data from Yfinance.

        1. Download data from yfinance
        2. Handle the correct symbol.
        3. Check for missing values.
        4. Return a clean dataframe.
        """

        symbol = symbol or self.config['symbol']

        self.logger.info(f"Fetching data for {symbol} from {self.config['start_date']} to {self.config['end_date']}")

        try:
            data = yf.download(symbol, start = self.config['start_date'],
                                end = self.config['end_date'],
                                progress = False)
            
        except Exception as e:

            self.logger.error(f"Failed to download {symbol}.")
            raise


        if data.empty:

            self.logger.error(f"No data fetched for {symbol}.")
            raise ValueError(f"No data fetched for {symbol}. Check date range.")

        missing = data.isnull().sum().sum()

        if missing > 0:
            
            self.logger.warning(f"Found {missing} missing values.")
            self.logger.debug(f"Missing values by column:\n{data.isnull().sum()}")

            initial_length = len(data)
            data = data.dropna() # Cleaning data
            self.logger.info(f"Dropped {initial_length - len(data)} rows with NaN values.")

        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

        missing_cols = [col for col in required_cols if col not in data.columns]

        if missing_cols:

            self.logger.error(f"Missing required columns: {missing_cols}.")
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        self.logger.info(f"Successfully fetched {len(data)} trading days.")
        self.logger.info(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
        self.logger.debug(f"Columns: {data.columns.tolist()}")

        av_vol = data['Volume'].mean()
        price_min = data['Close'].min()
        price_max = data['Open'].max()

        self.logger.info(f"Average daily volume: {av_vol}.")
        self.logger.info(f"Prince range: Rs. {price_min} to Rs. {price_max}.")

        daily_returns = ((data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1)) * 100
        print(daily_returns)

        extreme_moves = daily_returns[daily_returns.abs() > 0.01]

        if len(extreme_moves) > 0:
            
            self.logger.warning(f"Found {len(extreme_moves)} days with >10% price moves.")
            self.logger.debug(f"Extreme move dates: {extreme_moves.index.tolist()}")

        zero_vol = data[data['Volume'] == 0]

        if zero_vol.empty:

            self.logger.warning(f"Found {len(zero_vol)} days with zero volume.")
            data = data[data['Volume'] > 0]
            self.logger.info(f"Removed zero volume days. Final shape: {data.shape}")
        
        self.logger.info("Data fetching completed successfully")

        return data