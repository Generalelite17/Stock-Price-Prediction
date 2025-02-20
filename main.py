import os  # File operations
import logging
import yfinance as yf  # Fetch stock market data
import pandas as pd  # Handle tabular data
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split  # Split data into training/testing sets
from sklearn.ensemble import RandomForestClassifier # Machine learning model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report # Model evaluation
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt  # Plotting
import seaborn as sns  # Pretty visualizations
import xgboost as xgb 

import pandas as pd
import numpy as np

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

# Data Collection
def fetch_stock_data(ticker, start_date, end_date):

    """
    Fetch historical stock data from Yahoo Finance.

    Parameters:
    - ticker (str): The stock ticker symbol (e.g., "AAPL").
    - start_date (str): The start date in "YYYY-MM-DD" format.
    - end_date (str): The end date in "YYYY-MM-DD" format.

    Returns:
    - pd.DataFrame: A DataFrame containing historical stock data.
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Feature Engineering
def feature_engineering(stock_data):
 
 """
    Perform feature engineering on stock data.

    Computes technical indicators including moving averages, exponential moving averages,
    RSI, MACD, Bollinger Bands, volume moving averages, momentum, and lag features.
    Also calculates the target variable as a binary indicator of whether the next day's 
    closing price is higher than the current day's.

    Parameters:
    - stock_data (pd.DataFrame): DataFrame with stock data columns such as 'Date', 'Close',
      'High', 'Low', 'Open', and 'Volume'.

    Returns:
    - pd.DataFrame: The input DataFrame augmented with the new features, with NaN rows dropped.
    """
 
    # Moving Averages
    stock_data["MA5"] = stock_data["Close"].rolling(window=5).mean()
    stock_data["MA10"] = stock_data["Close"].rolling(window=10).mean()
    stock_data["MA50"] = stock_data["Close"].rolling(window=50).mean()
    
    # Exponential Moving Averages
    stock_data["EMA12"] = stock_data["Close"].ewm(span=12, adjust=False).mean()
    stock_data["EMA26"] = stock_data["Close"].ewm(span=26, adjust=False).mean()

    # Exponential Moving Averages
    stock_data["EMA12"] = stock_data["Close"].ewm(span=12, adjust=False).mean()
    stock_data["EMA26"] = stock_data["Close"].ewm(span=26, adjust=False).mean()

    # Daily Returns
    stock_data["Daily_Return"] = stock_data["Close"].pct_change()

    #Volatility (Rolling Standard Deviation)
    stock_data["Volatility"] = stock_data["Daily_Return"].rolling(window=21).std()

    # Relative Strength Index (RSI) - 14 periods
    delta = stock_data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    stock_data["RSI"] = 100 - (100 / (1 + rs))

    # Moving Average Convergence Divergence (MACD)
    stock_data["MACD"] = stock_data["EMA12"] - stock_data["EMA26"]
    stock_data["Signal_Line"] = stock_data["MACD"].ewm(span=9, adjust=False).mean()

    print(type(stock_data["Close"]))  # This should print <class 'pandas.core.series.Series'>
    print(stock_data["Close"].head())  # This will give you a preview of the data

    # Bollinger Bands (20-day SMA and 2 standard deviations)
    stock_data["SMA20"] = stock_data["Close"].rolling(window=20).mean()
    print(stock_data["Close"].rolling(window=20).std().head())

    print(stock_data.isna().sum())
    print(type(stock_data["Close"]))
    print(stock_data["Close"].dtypes)

    # Bollinger Bands (20-day SMA and 2 standard deviations)
    stock_data["Upper_Band"] = stock_data["SMA20"] + (stock_data.loc[:, "Close"].rolling(window=20).std() * 2)
    stock_data["Lower_Band"] = stock_data["SMA20"] - (stock_data.loc[:, "Close"].rolling(window=20).std() * 2)

    # Bollinger Band Width 
    stock_data["Bollinger_Width"] = stock_data["Upper_Band"] - stock_data["Lower_Band"]

    # Volume-related features (Rolling Average of Volume)
    stock_data["Volume_MA10"] = stock_data["Volume"].rolling(window=10).mean()

    # Momentum (5-day percentage change)
    stock_data["Momentum"] = stock_data["Close"].pct_change(periods=5)
    
    # Lag Features for Close and Volume (1, 2, and 3 days lag)
    for lag in [1, 2, 3]:
        stock_data[f'Close_lag{lag}'] = stock_data["Close"].shift(lag)
        stock_data[f'Volume_lag{lag}'] = stock_data["Volume"].shift(lag)

    # Target - 1 if price increased the next day, 0 if it decreased
    stock_data["Target"] = (stock_data["Close"].shift(-1) > stock_data["Close"]).astype(int)

    # Drop rows with NaN values after feature engineering
    return stock_data.dropna()

# Evaluation
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Down", "Up"], yticklabels=["Down", "Up"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    labels =  ["Down", "Up"]
    print(classification_report(y_true, y_pred, target_names=labels))

# Logging Configuration
logging.basicConfig(level=logging.CRITICAL, format= '%(asctime)s -%(levelname)s-%(message)s')

def run_project():
    logging.debug("Starting project...")  # Debugging print

    # Create data directory
    if not os.path.exists("data"):
        os.makedirs("data")
        logging.debug("Data folder does not exist. Data folder created.")  # Debugging print - Confirms no folder exist. Creates folder
    else:
        logging.debug("Data folder already exists") # Debuggin print - Confirms data folder exists and relays that to the terminal.

    # Data Collection
    #ticker = "AAPL"
    #Today = datetime.today().strftime("%Y-%m-%d")
    #stock_data = fetch_stock_data(ticker, "2020-01-01", Today)
    #logging.debug("Stock Data Downloaded!")  # Debugging print
    #print("Raw data shape:", stock_data.shape)
    #print(stock_data.head())

    dummy_data.to_csv("data/AAPL_data.csv")

    # stock_data.columns = stock_data.columns.droplevel(0)  # Drop the first level (Price)

    # # Rename the columns to reflect the actual price types
    # stock_data.columns = ['Date','Close', 'High', 'Low', 'Open', 'Volume']

    # stock_data.columns.name = None  # Remove the name of columns
    
    # stock_data.to_csv("data/AAPL_data_1.csv")

    # print(stock_data.head())  # Check first few rows
    # print(stock_data.columns)  # Check column headers

    # Feature Engineering
    processed_data = feature_engineering(dummy_data)
    processed_data.to_csv("data/processed_data.csv", index=False)
    logging.debug("Data processed and saved!")  # Debugging print

   # Select features and target
    features = [
        "MA5", "MA10", "MA50", "EMA12", "EMA26", "Daily_Return", "Volatility",
        "RSI", "MACD", "Signal_Line", "SMA20", "Upper_Band", "Lower_Band",
        "Bollinger_Width", "Volume_MA10", "Momentum",
        "Close_lag1", "Close_lag2", "Close_lag3",
        "Volume_lag1", "Volume_lag2", "Volume_lag3"
    ]
    X = processed_data[features]
    y = processed_data["Target"]
    
    # Split data (without shuffling for time series data)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Initialize the XGBoost classifier
    xgb_model = xgb.XGBClassifier(
        random_state=42, 
        #use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    
    # Use TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search on the training data
    grid_search.fit(X_train, y_train)
    print("Best parameters found:", grid_search.best_params_)
    
    # Use the best estimator to predict on the test set
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"XGBoost Model Accuracy: {accuracy:.2f}")

   # Call the custom confusion matrix plotting function
    plot_confusion_matrix(y_test, predictions)

if __name__ == "__main__":
    run_project()
