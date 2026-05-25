"""food_index_calculator.py — Core engine for computing bottom-up Food Price Index."""
import pandas as pd
import numpy as np
from src.utils import get_logger
from src.inflation.index_weights import load_weights_map

logger = get_logger(__name__)

class FoodIndexCalculator:
    """Calculates bottom-up Food Price Index using Laspeyres index formula
    with base year 2021, mapping weights and handling subset forecasts.
    """
    def __init__(self, historical_df: pd.DataFrame):
        """Initializes the calculator by computing the base prices of items in 2021.
        
        Args:
            historical_df: DataFrame with DatetimeIndex and food price columns.
        """
        self.historical_df = historical_df.copy()
        
        # Load weights mapped to columns
        self.weights = load_weights_map(price_columns=self.historical_df.columns)
        self.food_items = list(self.weights.keys())
        
        if not self.food_items:
            logger.error("No food items matched with weights. Index calculation will fail.")
            raise ValueError("No food items found matching base period weights.")
            
        # Compute base year 2021 average prices
        self._compute_base_prices()

    def _compute_base_prices(self):
        """Computes average prices for base year 2021 for all items in the basket."""
        # Find 2021 mask
        mask_2021 = self.historical_df.index.year == 2021
        
        if mask_2021.any():
            base_prices = self.historical_df.loc[mask_2021, self.food_items].mean()
        else:
            logger.warning("No data found for base year 2021. Using overall dataset mean.")
            base_prices = self.historical_df[self.food_items].mean()
            
        # Handle zero/missing prices to avoid division by zero
        self.base_prices = base_prices.replace(0, np.nan).fillna(self.historical_df[self.food_items].mean()).fillna(1.0)
        logger.info("Base prices for year 2021 computed successfully.")

    def calculate_historical_index(self) -> pd.Series:
        """Reconstructs the index historically using the same Laspeyres formula.
        
        Returns:
            pd.Series: Calculated historical index with DatetimeIndex.
        """
        df_matched = self.historical_df[self.food_items]
        w_series = pd.Series(self.weights)
        
        # Laspeyres index: Sum_i (w_i * Price_it / Price_i0) * 100
        comp = (df_matched.div(self.base_prices, axis=1) * w_series).sum(axis=1) * 100
        comp.name = "Computed_Index"
        return comp

    def get_future_prices(self, results: dict, horizon: int) -> pd.DataFrame:
        """Constructs the future price matrix, combining forecasts with constant baselines.
        
        Args:
            results: Results dictionary containing forecast DataFrames for each target.
            horizon: Forecast horizon (months).
            
        Returns:
            pd.DataFrame: Matrix of shape (horizon, num_items) with DatetimeIndex.
        """
        last_date = self.historical_df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq='MS')
        
        # Build price DataFrame for the future months
        future_prices = pd.DataFrame(index=future_dates, columns=self.food_items)
        
        for item in self.food_items:
            has_forecast = False
            if item in results and "forecasts" in results[item] and horizon in results[item]["forecasts"]:
                fc_df = results[item]["forecasts"][horizon]
                fc_col = f"{item} (Forecast)"
                if fc_col in fc_df.columns:
                    # Align forecasts by date/month
                    fc_df_temp = fc_df.copy()
                    fc_df_temp.index = pd.to_datetime(fc_df_temp["Month"])
                    future_prices[item] = fc_df_temp[fc_col]
                    has_forecast = True
                    
            if not has_forecast:
                # Naive fallback: Hold price constant at the last known historical value
                last_price = self.historical_df[item].iloc[-1]
                future_prices[item] = last_price
                
        # Fill any missing values
        return future_prices.ffill().bfill().fillna(1.0)

    def calculate_future_index(self, results: dict, horizon: int) -> pd.DataFrame:
        """Calculates the future index dynamically from predictions.
        
        If a food item was not selected for forecasting, its price is held constant
        at its last known historical value.
        
        Args:
            results: Results dictionary containing forecast DataFrames for each target.
            horizon: Forecast horizon (months).
            
        Returns:
            pd.DataFrame: DataFrame containing columns 'Month' and 'Index (Forecast)'.
        """
        future_prices = self.get_future_prices(results, horizon)
        
        # Apply Laspeyres formula
        w_series = pd.Series(self.weights)
        calculated_index = (future_prices.div(self.base_prices, axis=1) * w_series).sum(axis=1) * 100
        
        # Build output DataFrame matching the pipeline structure
        future_months = future_prices.index.strftime("%Y-%m").tolist()
        out_df = pd.DataFrame({
            "Month": future_months,
            "Index (Forecast)": calculated_index.values.round(4)
        })
        
        logger.info(f"Calculated future food index for {horizon}m horizon: {out_df['Index (Forecast)'].tolist()}")
        return out_df

