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
