# Stock-Price-Prediction

## Overview
This project predicts the movement of stock prices (up or down) using historical data from **Yahoo Finance**. It leverages **machine learning models** such as **logistic regression** and **random forests** to classify whether the closing price will increase the next day.  

By building this, I aim to showcase my skills in **data science, machine learning, and financial analysis** while expanding my knowledge of quantitative finance.  

---

## Technologies Used
- **Python**  
- **Yahoo Finance API (`yfinance`)** – For fetching historical stock prices  
- **Pandas & NumPy** – For data manipulation  
- **Scikit-learn** – For machine learning models  
- **Matplotlib & Seaborn** – For data visualization  

---

##  Data Collection
The dataset is retrieved from **Yahoo Finance**, containing the following columns:  

- `Open`, `High`, `Low`, `Close` prices  
- `Volume` (trading volume)  
- `Moving Averages (MA10, MA50)`  
- `Daily Returns`  
- `Target Variable` (1 = price goes up, 0 = price goes down)  

Sample retrieval code:  
```python
import yfinance as yf

# Download historical data for a given stock
ticker = "AAPL"
stock_data = yf.download(ticker, start="2020-01-01", end="2025-01-01")
print(stock_data.head())

stock_data["MA10"] = stock_data["Close"].rolling(window=10).mean()
stock_data["MA50"] = stock_data["Close"].rolling(window=50).mean()
stock_data["Daily_Return"] = stock_data["Close"].pct_change()
stock_data["Target"] = (stock_data["Close"].shift(-1) > stock_data["Close"]).astype(int)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Prepare data
features = ["MA10", "MA50", "Daily_Return"]
X = stock_data[features].dropna()
y = stock_data.loc[X.index, "Target"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model Accuracy: {accuracy:.2f}")

git clone https://github.com/yourusername/Stock-Price-Prediction.git
cd Stock-Price-Prediction

pip install -r requirements.txt

📁 Stock-Price-Prediction
 ├── 📄 stock_prediction.py  # Main script
 ├── 📊 stock_data.csv       # Sample dataset (optional)
 ├── 📄 README.md            # Project documentation
 ├── 📜 requirements.txt     # Required dependencies
 ├── 📂 notebooks/           # Jupyter notebooks
 ├── 📊 images/              # Visualizations

🏆 About Me
I am a mechanical engineer with a master's in computer science, passionate about quantitative finance, AI, and software engineering. This project reflects my curiosity about financial markets and my technical expertise in data-driven decision-making.





