import pandas as pd
import numpy as np

# Dummy Data - Randomly generate data if user has any issue with Yahoo Finace API
# Create a date range
dates = pd.date_range(start="2020-01-01", periods=10000, freq='D')

# Generate dummy stock data
dummy_data = pd.DataFrame({
    'Date': dates,
    'Close': np.random.uniform(100, 150, size=10000),
    'High': np.random.uniform(100, 150, size=10000),
    'Low': np.random.uniform(100, 150, size=10000),
    'Open': np.random.uniform(100, 150, size=10000),
    'Volume': np.random.randint(100000, 200000, size=10000)
})
print("Dummy data shape:", dummy_data.shape)
print(dummy_data.head())
