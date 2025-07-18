import pandas as pd
import yfinance as yf
from datetime import datetime
import numpy as np
import streamlit as st
import requests
import json

PERPLEXITY_API_KEY = "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"
API_ENDPOINT = "https://api.perplexity.ai/chat/completions"

def call_perplexity_api(messages, model="sonar-reasoning"):
    # Strict validation
    if not isinstance(messages, list):
        raise ValueError(f"messages must be a list, got {type(messages)}: {messages}")
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"Message {i} is not dict: {msg}")
        if "role" not in msg or "content" not in msg:
            raise ValueError(f"Message {i} missing keys: {msg}")
        # Coerce all content to string and strip
        msg["role"] = str(msg["role"]).strip()
        msg["content"] = str(msg["content"]).strip()
        if not msg["role"]:
            raise ValueError(f"Role in message {i} is blank")
        if not msg["content"]:
            raise ValueError(f"Content in message {i} is blank")
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY.strip()}",
        "Content-Type": "application/json",
        "accept": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages
    }
    print("\nSENDING THIS PAYLOAD TO PERPLEXITY API:")
    print(json.dumps(payload, indent=2))
    try:
        response = requests.post(API_ENDPOINT, headers=headers, data=json.dumps(payload))
    except Exception as e:
        print(f"Request exception: {e}")
        raise
    if response.status_code != 200:
        print("Status code:", response.status_code)
        print("Response text:", response.text)
        try:
            print("Response JSON:", response.json())
        except Exception:
            print("Non-JSON response:", response.text)
        response.raise_for_status()
    r = response.json()
    if not r.get("choices"):
        print("API returned no choices. Full response:", r)
        return "No response."
    return r["choices"][0]["message"]["content"]

# ==== Finance utility functions ====
def get_basic_stock_info(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    basic_info = pd.DataFrame({
        'Name': [info.get('longName', 'N/A')],
        'Sector': [info.get('sector', 'N/A')],
        'Industry': [info.get('industry', 'N/A')],
        'Market Cap': [info.get('marketCap', 'N/A')],
        'Current Price': [info.get('currentPrice', 'N/A')],
        '52 Week High': [info.get('fiftyTwoWeekHigh', 'N/A')],
        '52 Week Low': [info.get('fiftyTwoWeekLow', 'N/A')],
        'Average Volume': [info.get('averageVolume', 'N/A')]
    })
    return basic_info

def get_fundamental_analysis(ticker, period='1y'):
    stock = yf.Ticker(ticker)
    history = stock.history(period=period)
    info = stock.info
    return pd.DataFrame({
        'PE Ratio': [info.get('trailingPE', 'N/A')],
        'Forward PE': [info.get('forwardPE', 'N/A')],
        'PEG Ratio': [info.get('pegRatio', 'N/A')],
        'Price to Book': [info.get('priceToBook', 'N/A')],
        'Dividend Yield': [info.get('dividendYield', 'N/A')],
        'EPS (TTM)': [info.get('trailingEps', 'N/A')],
        'Revenue Growth': [info.get('revenueGrowth', 'N/A')],
        'Profit Margin': [info.get('profitMargins', 'N/A')],
        'Free Cash Flow': [info.get('freeCashflow', 'N/A')],
        'Debt to Equity': [info.get('debtToEquity', 'N/A')],
        'Return on Equity': [info.get('returnOnEquity', 'N/A')],
        'Operating Margin': [info.get('operatingMargins', 'N/A')],
        'Quick Ratio': [info.get('quickRatio', 'N/A')],
        'Current Ratio': [info.get('currentRatio', 'N/A')],
        'Earnings Growth': [info.get('earningsGrowth', 'N/A')],
        'Stock Price Avg (Period)': [history['Close'].mean()],
        'Stock Price Max (Period)': [history['Close'].max()],
        'Stock Price Min (Period)': [history['Close'].min()]
    })

def get_stock_risk_assessment(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    history = stock.history(period=period)
    if history.empty:
        return pd.DataFrame({
            'Annualized Volatility': [np.nan],
            'Beta': [np.nan],
            'Value at Risk (95%)': [np.nan],
            'Maximum Drawdown': [np.nan],
            'Sharpe Ratio': [np.nan],
            'Sortino Ratio': [np.nan]
        })

    returns = history['Close'].pct_change().dropna()
    # ... rest of your calculations

def calculate_beta(stock_returns, market_ticker, period):
    market = yf.Ticker(market_ticker)
    market_history = market.history(period=period)
    market_returns = market_history['Close'].pct_change().dropna()
    aligned_returns = pd.concat([stock_returns, market_returns], axis=1).dropna()
    covariance = aligned_returns.cov().iloc[0, 1]
    market_variance = market_returns.var()
    return covariance / market_variance

def calculate_max_drawdown(prices):
    peak = prices.cummax()
    drawdown = (prices - peak) / peak
    return drawdown.min()

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate/252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_sortino_ratio(returns, risk_free_rate=0.02, target_return=0):
    excess_returns = returns - risk_free_rate/252
    downside_returns = excess_returns[excess_returns < target_return]
    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    return np.sqrt(252) * excess_returns.mean() / downside_deviation

def get_technical_analysis(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    history = stock.history(period=period)
    
    # Early exit if no data
    if history.empty:
        return pd.DataFrame({
            'Indicator': [
                'Current Price',
                '50-day SMA',
                '200-day SMA',
                'RSI (14-day)',
                'MACD',
                'MACD Signal'
            ],
            'Value': ['N/A'] * 6
        })
    
    history['SMA_50'] = history['Close'].rolling(window=50).mean()
    history['SMA_200'] = history['Close'].rolling(window=200).mean()
    history['RSI'] = calculate_rsi(history['Close'])
    history['MACD'], history['Signal'] = calculate_macd(history['Close'])
    latest = history.iloc[-1]
    return pd.DataFrame({
        'Indicator': [
            'Current Price',
            '50-day SMA',
            '200-day SMA',
            'RSI (14-day)',
            'MACD',
            'MACD Signal'
        ],
        'Value': [
            f'${latest["Close"]:.2f}',
            f'${latest["SMA_50"]:.2f}',
            f'${latest["SMA_200"]:.2f}',
            f'{latest["RSI"]:.2f}',
            f'{latest["MACD"]:.2f}',
            f'{latest["Signal"]:.2f}'
        ]
    })


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, short_window=12, long_window=26, signal_window=9):
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def get_stock_news(ticker, limit=5):
    stock = yf.Ticker(ticker)
    news = stock.news[:limit]
    news_data = []
    for article in news:
        news_entry = {
            "Title": article['title'],
            "Publisher": article['publisher'],
            "Published": datetime.fromtimestamp(article['providerPublishTime']).strftime('%Y-%m-%d %H:%M:%S'),
            "Link": article['link']
        }
        news_data.append(news_entry)
    return pd.DataFrame(news_data)

# ==== LLM orchestration functions ====

def identify_stock_and_query(query):
    messages = [
        {"role": "system", "content": "Extract the ticker(s), timeframe (if possible), user expertise, and analysis focus from the following user stock analysis question."},
        {"role": "user", "content": str(query)}
    ]
    return call_perplexity_api(messages)

def summarize_basic_info(basic_info, query):
    table_str = basic_info.to_markdown()
    messages = [
        {"role": "system", "content": "You are a junior stock researcher. Present the following stock financial table as a concise summary relevant for the user query."},
        {"role": "user", "content": f"Query: {query}\n\n{table_str}"}
    ]
    return call_perplexity_api(messages)

def summarize_analysis(fundamental, technical, risk, query):
    fundamental_str = fundamental.to_markdown()
    technical_str = technical.to_markdown()
    risk_str = risk.to_markdown()
    messages = [
        {"role": "system", "content": "You are a seasoned financial analyst. Write a detailed but crisp analysis relevant to the user query using the provided data tables."},
        {"role": "user", "content": f"Query: {query}\n\nFundamental data:\n{fundamental_str}\n\nTechnical data:\n{technical_str}\n\nRisk metrics:\n{risk_str}"}
    ]
    return call_perplexity_api(messages)

def summarize_news(news_df, query):
    news_str = news_df.to_markdown() if not news_df.empty else 'No news.'
    messages = [
        {"role": "system", "content": "You are a financial news analyst. Summarize the latest news headlines in terms of their likely impact on the stock's performance, citing positive or negative sentiment for the user query."},
        {"role": "user", "content": f"Query: {query}\n\nNews list:\n{news_str}"}
    ]
    return call_perplexity_api(messages)

def generate_final_report(query, basic_summary, analysis_summary, news_summary):
    messages = [
        {"role": "system", "content": "You are an expert financial report writer. Combine the following sections into a single, professional markdown stock report. Start with an executive summary answering the user query, and end with a clear, confident investment recommendation. Omit redundancy."},
        {"role": "user", "content": f"User question: {query}\n\nBasic info summary:\n{basic_summary}\n\nAnalysis section:\n{analysis_summary}\n\nNews section:\n{news_summary}"}
    ]
    return call_perplexity_api(messages)

# ==== Streamlit UI ====
st.set_page_config(page_title="Advanced Stock Analysis Dashboard", layout="wide")
st.title("Advanced Stock Analysis Dashboard")

st.sidebar.header("Stock Analysis Query")
query = st.sidebar.text_area("Enter your stock analysis question", "Is Tata Motors a safe long-term bet for a risk-averse individual?", height=100)
analyze_button = st.sidebar.button("Analyze")

if analyze_button:
    st.info("Analyzing...")
    extraction = identify_stock_and_query(query)
    st.markdown("### Query Interpretation")
    st.markdown(extraction)

    ticker = None
    for w in extraction.split():
        if w.isupper() and (len(w) <= 6 or '.' in w):
            ticker = w.strip(".:,;")
    if not ticker:
        st.error("Unable to extract ticker from LLM response.\nRaw extraction:\n" + extraction)
        st.stop()

    basic_info_df = get_basic_stock_info(ticker)
    fundamental_df = get_fundamental_analysis(ticker)
    risk_df = get_stock_risk_assessment(ticker)
    technical_df = get_technical_analysis(ticker)
    news_df = get_stock_news(ticker, limit=7)

    basic_summary = summarize_basic_info(basic_info_df, query)
    analysis_summary = summarize_analysis(fundamental_df, technical_df, risk_df, query)
    news_summary = summarize_news(news_df, query)

    report = generate_final_report(query, basic_summary, analysis_summary, news_summary)

    st.success("Analysis complete!")
    st.markdown("## Full Analysis Report")
    st.markdown(report)

st.markdown("---")
st.markdown("Made by Chandra")

