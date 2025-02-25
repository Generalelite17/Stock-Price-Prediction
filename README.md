# 📈 Stock Price Movement Prediction

## Overview
This project is designed to predict the daily movement of stock prices—specifically, whether the closing price will increase (Up) or decrease (Down) the next day. The pipeline includes data collection, feature engineering to compute technical indicators, model training with XGBoost, and evaluation using accuracy metrics, confusion matrices, and classification reports.

*Note: For development and testing purposes, dummy data is used to simulate stock market data. The pipeline is structured to easily switch to real data from Yahoo Finance once API rate-limiting issues are resolved.*

---

## Technologies Used
- **Python**
- **Pandas & NumPy** – Data manipulation and numerical operations.
- **yfinance** – For fetching historical stock data (to be used when rate limits are resolved).
- **Scikit-learn** – For data splitting, model evaluation, and hyperparameter tuning.
- **XGBoost** – Machine learning model used for prediction.
- **Matplotlib & Seaborn** – For data visualization (confusion matrix, plots, etc.).
- **Logging** – To track and debug the pipeline's execution.

---

## Data Collection & Preprocessing
The project is designed to work with data containing the following columns:
- **Date, Open, High, Low, Close, Volume**

For now, dummy data is generated using the following structure:
- A date range starting from 2020-01-01.
- Randomly generated values for stock prices (`Open`, `High`, `Low`, `Close`) and `Volume`.

### Feature Engineering
The pipeline computes several technical indicators:
- **Moving Averages (MA5, MA10, MA50)**
- **Exponential Moving Averages (EMA12, EMA26)**
- **Daily Returns and Volatility**
- **Relative Strength Index (RSI)**
- **MACD and Signal Line**
- **Bollinger Bands and Bollinger Band Width**
- **Volume Moving Average (Volume_MA10)**
- **Momentum (5-day percentage change)**
- **Lag Features for both Close and Volume (lag 1, 2, and 3 days)**
- **Target Variable:** Binary indicator (1 if next day's close > today's close, else 0).

---

## Project Structure
```
Stock-Price-Prediction/
├── data/
│   ├── raw_data.csv              # Raw stock data (dummy or real API data)
│   ├── unprocessed_data.csv      # Data after initial preprocessing (e.g., dropping extra headers)
│   └── processed_data.csv        # Data after full feature engineering
│
├── dummy_data_generation.py      # Script for generating dummy stock data
├── yf_api_connection_test.py     # Script for testing Yahoo Finance API connectivity
├── main.py                       # Main script that runs the entire pipeline
├── README.md                     # Project overview and documentation
├── requirements.txt              # List of dependencies with version specifications
└── LICENSE                       # License file (e.g., MIT License)


---

## How to Run the Project
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Stock-Price-Prediction.git
   cd Stock-Price-Prediction
   ```

2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the main script:**
   ```bash
   python main.py
   ```

*Note: The current setup usese real data fetched from the Yahoo Finance API. However, if needed,  you can switch to generated dummy data for testing by uncommenting the dummy data for testing by uncommenting the dummy data collection code in `main.py`.*

---

## Model Evaluation
The project uses XGBoost with hyperparameter tuning via grid search and time-series cross-validation. The model's performance is evaluated using:
- **Accuracy**
- **Confusion Matrix:** Visualized with Seaborn heatmaps.
- **Classification Report:** Providing precision, recall, F1-score, and support for each class.

Example output on dummy data:
- **Accuracy:** ~73%
- **Confusion Matrix & Classification Report:** Displayed in the console and as a plot.

Example Output on Real Data:
- **Accuracy:** ~75%

Confusion Matrix & Classification Report:
- The confusion matrix and classification report are displayed both in the console and as a plot. For example, you might observe that:
- The Down class has high recall (e.g., 0.75–0.99).
- The Up class shows very low recall (e.g., 0.00–0.14) and an F1-score near 0.
- This suggests that while overall accuracy is around 75%, the model is biased toward predicting the "Down" class.

Potential Issues:

- **Class Imbalance:** The target distribution (e.g., 901 Down vs. 343 Up) can cause the model to favor the majority class, leading to poor performance for the "Up" class.
- **Target Definition:** The binary target (1 if next-day closing price increases by at least 1%, else 0) might capture noise or fail to effectively differentiate meaningful upward moves.
- **Feature Informativeness:** The current technical indicators may not be sufficient for the model to reliably predict upward movements.

---

## Future Enhancements
- **Real Data Integration:** Resolve rate-limiting issues with Yahoo Finance or use an alternative data provider (e.g., Alpha Vantage, IEX Cloud) to fetch real stock data.
- **Hyperparameter Optimization:** Experiment with additional hyperparameters or alternative models (e.g., LSTM networks) to improve performance.
- **Deployment:** Consider deploying the model as an API using Flask/FastAPI or building a dashboard with Streamlit or Dash for live predictions.
- **Documentation & Testing:** Expand unit tests and documentation for each module to improve maintainability and facilitate future development.

---

## License
This project is licensed under the [MIT License](LICENSE).
