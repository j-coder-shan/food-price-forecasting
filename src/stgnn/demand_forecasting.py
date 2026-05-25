import numpy as np
import pandas as pd
from src.utils import get_food_columns

class DemandForecaster:
    """
    Generates and manages Synthetic Demand Scores for food commodities.
    In the real world, this would load actual market sales data.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.food_cols = get_food_columns(self.df)
        
    def generate_synthetic_demand(self) -> pd.DataFrame:
        """
        Generates demand scores based on:
        1. Base popularity per food group.
        2. Seasonality (festivals in April, December).
        3. Price elasticity (higher price -> slightly lower demand, unless staple).
        4. General market inflation trends.
        """
        demand_df = pd.DataFrame(index=self.df.index)
        
        # Festival months in Sri Lanka (April: New Year, December: Christmas)
        month_series = self.df.index.month
        festival_boost = np.where(month_series.isin([4, 12]), 15.0, 0.0)
        
        for col in self.food_cols:
            # Base demand (random base popularity between 30 and 80)
            np.random.seed(hash(col) % (2**32))
            base_demand = np.random.uniform(40, 70)
            
            # Staple modifier (rice, bread, dhal are staples -> inelastic)
            is_staple = any(staple in col.lower() for staple in ['rice', 'bread', 'dhal', 'sugar', 'wheat'])
            elasticity = 0.1 if is_staple else 0.4
            
            # Price history
            prices = self.df[col].fillna(method='bfill').fillna(method='ffill').values
            if len(prices) == 0 or np.isnan(prices).all():
                prices = np.zeros(len(self.df))
                
            # Rolling baseline for price shock
            rolling_price = pd.Series(prices).rolling(window=6, min_periods=1).mean().values
            price_shock = (prices - rolling_price) / (rolling_price + 1e-5)
            
            # Demand calculation
            demand = base_demand + festival_boost - (price_shock * 100 * elasticity)
            
            # Add some random noise
            noise = np.random.normal(0, 3, len(demand))
            demand += noise
            
            # Clip between 0 and 100 (Demand Score Index)
            demand_df[f"{col}_Demand"] = np.clip(demand, 0, 100)
            
        return demand_df
