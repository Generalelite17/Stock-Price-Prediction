def run_project():
    import yfinance as yf
    ticker = "MSFT"
    data = yf.download(ticker, start="2020-01-01", end="2023-01-01")
    print(data.shape)
    print(data.head())

if __name__ == "__main__":
    run_project()
