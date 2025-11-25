from flask import Flask, render_template, jsonify
import yfinance as yf
import pandas as pd
import sys
import argparse
import urllib.request
import urllib.error
import json
import re
import time
import logging
import warnings
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Set
import random
from datetime import datetime, timedelta
import os
import requests
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import feedparser

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress statsmodels ValueWarnings about date frequency
warnings.filterwarnings('ignore', message='.*date index.*frequency.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*A date index has been provided.*', category=UserWarning)

# Try to import statsmodels for ARIMA and Granger causality
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller, grangercausalitytests
    from statsmodels.stats.stattools import durbin_watson
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("statsmodels not available - ARIMA and Granger causality functionality will use fallback")

# Try to import arch for GARCH
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    logger.warning("arch library not available - GARCH functionality will use rolling volatility fallback")

# Try to import sklearn for PCA and MDS
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import MDS
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available - PCA and MDS functionality will use fallback")

# Try to import scipy for distance computations in MDS
try:
    from scipy.spatial.distance import pdist, squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available - MDS distance computation will use fallback")

# ================== NewsAPI + FinBERT Sentiment Config ==================

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")  # must be set in environment

FINBERT_MODEL_NAME = "ProsusAI/finbert"

FINBERT_LABELS = ["negative", "neutral", "positive"]  # default label order for this checkpoint

try:
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
    finbert_model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
    FINBERT_AVAILABLE = True
    logger.info("FinBERT model loaded successfully")
except Exception as e:
    FINBERT_AVAILABLE = False
    finbert_model = None
    tokenizer = None
    logger.error(f"Failed to load FinBERT model: {e}")

# Sentiment engine mode: "newsapi_finbert" or "mock"
SENTIMENT_ENGINE_MODE = "newsapi_finbert"

def finbert_sentiment_score(text: str) -> float:
    """
    Use FinBERT to compute a scalar sentiment score in [-1, 1] for the given text.
    negative ≈ -1, neutral ≈ 0, positive ≈ +1, weighted by confidence.
    Returns 0.0 if FinBERT is unavailable or any error occurs.
    """
    try:
        if not FINBERT_AVAILABLE or finbert_model is None or tokenizer is None:
            logger.warning("FinBERT not available, returning neutral sentiment")
            return 0.0
        
        if not text:
            return 0.0
        
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        )
        
        with torch.no_grad():
            outputs = finbert_model(**encoded)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)[0].cpu().tolist()
        
        # argmax gives class index
        max_idx = int(torch.argmax(logits, dim=-1)[0])
        
        label_to_value = {
            0: -1.0,  # negative
            1: 0.0,   # neutral
            2: 1.0    # positive
        }
        base_val = label_to_value.get(max_idx, 0.0)
        max_prob = probs[max_idx]
        
        compound = base_val * max_prob
        return float(compound)
    except Exception as e:
        logger.error(f"Error in finbert_sentiment_score: {e}", exc_info=True)
        return 0.0

app = Flask(__name__)

# Cache mechanism to avoid hitting rate limits
cache = {}
cache_timeout = 300  # 5 minutes

# Valid yfinance periods
VALID_YF_PERIODS = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}

def normalize_period(raw: str, default: str = "1y") -> str:
    """
    Normalize period string to a valid yfinance period.
    Converts common invalid formats (e.g., "1m" -> "1mo") to valid ones.
    """
    if not raw:
        return default
    raw = raw.strip()
    if raw in VALID_YF_PERIODS:
        return raw
    mapping = {
        "1m": "1mo",
        "3m": "3mo",
        "6m": "6mo",
    }
    return mapping.get(raw, default)

def get_cached_data(key, fetch_function):
    current_time = time.time()
    if key in cache and current_time - cache[key]['timestamp'] < cache_timeout:
        return cache[key]['data']
    
    try:
        data = fetch_function()
        cache[key] = {
            'data': data,
            'timestamp': current_time
        }
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        # Return mock data if real data fetch fails
        return get_mock_data(key)

def get_mock_data(key):
    # Mock data for different endpoints
    if key == 'market_summary':
        return {
        'nifty_data': {
            'dates': [str(datetime.now() - timedelta(days=i)) for i in range(30, 0, -1)],
            'prices': [random.uniform(22000, 24500) for _ in range(30)]
        },
        'sentiment': random.choice(['Bullish', 'Bearish', 'Neutral']),
        'market_status': random.choice(['Open', 'Closed']),
        'nifty_change': f"{random.uniform(-1.5, 1.5):.2f}%",
        'india_vix': f"{random.uniform(10, 20):.2f}"
    }

    return {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/time_series_analysis')
def time_series_analysis():
    return render_template('time_series_analysis.html')

@app.route('/sector_analytics')
def sector_analytics():
    return render_template('sector_analytics.html')

@app.route('/volatility_ranking')
def volatility_ranking():
    return render_template('volatility_ranking.html')

# Flask route: market_summary — switch to NIFTY 50 (^NSEI) and India VIX (^INDIAVIX)
@app.route('/api/market_summary')
def market_summary():
    def fetch_data():
        try:
            nifty = yf.Ticker("^NSEI")
            hist = nifty.history(period="1mo")
            dates = hist.index.strftime('%Y-%m-%d').tolist()
            prices = hist['Close'].tolist()

            nifty_info = nifty.info
            nifty_change = nifty_info.get('regularMarketChangePercent', 0)

            india_vix = yf.Ticker("^INDIAVIX")
            vix_info = india_vix.info
            india_vix_value = vix_info.get('regularMarketPrice', 0)

            # ========== Calculate real sentiment from sector sentiment data ==========
            sentiment_score = 0.0
            sentiment_label = "Neutral"
            
            try:
                # Get sector sentiment data
                sector_map = get_nifty50_sector_mapping()
                
                # Try to get real sentiment from Google News + FinBERT
                if FINBERT_AVAILABLE:
                    articles = get_cached_data(
                        'google_news_nifty_market_summary',
                        lambda: fetch_google_news_headlines("Nifty 50")
                    )
                    
                    sector_sentiment = build_sector_sentiment_from_news_finbert(
                        articles,
                        sector_map
                    )
                    
                    if sector_sentiment:
                        # Aggregate all sector sentiments to get overall market sentiment
                        all_sentiment_scores = []
                        for sector, data in sector_sentiment.items():
                            if 'current_sentiment' in data:
                                all_sentiment_scores.append(data['current_sentiment'])
                        
                        if all_sentiment_scores:
                            sentiment_score = float(np.mean(all_sentiment_scores))
                            logger.info(f"Market sentiment from sector data: {sentiment_score:.3f}")
                
                # If no sentiment from news, use a combination of price action and VIX
                if sentiment_score == 0.0:
                    # Fallback: use price action and VIX to infer sentiment
                    if nifty_change > 1 and india_vix_value < 15:
                        sentiment_score = 0.3  # Slightly positive
                    elif nifty_change < -1 or india_vix_value > 20:
                        sentiment_score = -0.3  # Slightly negative
                    else:
                        sentiment_score = 0.0  # Neutral
                
                # Convert sentiment score to label
                if sentiment_score > 0.2:
                    sentiment_label = "Bullish"
                elif sentiment_score < -0.2:
                    sentiment_label = "Bearish"
                else:
                    sentiment_label = "Neutral"
                    
            except Exception as e:
                logger.warning(f"Error calculating sentiment from sector data: {e}, using fallback")
                # Fallback to simple rule-based sentiment
                if nifty_change > 1 and india_vix_value < 15:
                    sentiment_label = "Bullish"
                elif nifty_change < -1 or india_vix_value > 20:
                    sentiment_label = "Bearish"
                else:
                    sentiment_label = "Neutral"

            now = datetime.now()
            market_status = "Closed"
            # Simplified: typical NSE trading window approximation (local time)
            if now.weekday() < 5 and 9 <= now.hour < 16:
                market_status = "Open"

            return {
                'nifty_data': {'dates': dates, 'prices': prices},
                'sentiment': sentiment_label,
                'sentiment_score': sentiment_score,  # Include score for debugging
                'market_status': market_status,
                'nifty_change': f"{nifty_change:.2f}%",
                'india_vix': f"{india_vix_value:.2f}"
            }
        except Exception as e:
            logger.error(f"Error fetching Indian market summary: {e}", exc_info=True)
            return get_mock_data('market_summary')
    return jsonify(get_cached_data('market_summary', fetch_data))

# Gainers/Losers — use Indian large-cap list and actual change %
@app.route('/api/gainers_losers')
def gainers_losers():
    def fetch_data():
        # Expanded symbol list to get more real data
        symbols = [
            'RELIANCE.NS','TCS.NS','HDFCBANK.NS','INFY.NS','ITC.NS','SBIN.NS','BHARTIARTL.NS','ICICIBANK.NS',
            'KOTAKBANK.NS','LT.NS','HINDUNILVR.NS','AXISBANK.NS','ASIANPAINT.NS','MARUTI.NS','BAJFINANCE.NS',
            'WIPRO.NS','ULTRACEMCO.NS','NESTLEIND.NS','TITAN.NS','HDFCLIFE.NS','SUNPHARMA.NS','DRREDDY.NS',
            'EICHERMOT.NS','DIVISLAB.NS','POWERGRID.NS','GAIL.NS','BPCL.NS','HEROMOTOCO.NS','ADANIENT.NS',
            'ADANIPORTS.NS','JSWSTEEL.NS','TATASTEEL.NS','GRASIM.NS','HCLTECH.NS','TECHM.NS','M&M.NS','CIPLA.NS',
            'BRITANNIA.NS','BAJAJ-AUTO.NS','SHREECEM.NS','TATAMOTORS.NS','COALINDIA.NS','ONGC.NS','NTPC.NS',
            'LUPIN.NS','DLF.NS','SBILIFE.NS','ZOMATO.NS','PNB.NS','BHARATFORG.NS','MRF.NS','PETRONET.NS',
            'ICICIPRULI.NS','MINDTREE.NS','LTIM.NS','SRF.NS','PAGEIND.NS','VEDL.NS','INDUSINDBK.NS','BANKBARODA.NS',
            'CANBK.NS','UNIONBANK.NS','IDFCFIRSTB.NS','FEDERALBNK.NS','YESBANK.NS','RBLBANK.NS','AUBANK.NS',
            'HDFCAMC.NS','ICICIGI.NS','BAJAJFINSV.NS','MUTHOOTFIN.NS','CHOLAFIN.NS','SHRIRAMFIN.NS','LICHSGFIN.NS',
            'APOLLOTYRE.NS','CEATLTD.NS','BALKRISIND.NS','TVSMOTOR.NS','ESCORTS.NS','ASHOKLEY.NS',
            'MAHINDRA.NS'
        ]

        gainers_list = []
        losers_list = []
        
        try:
            # Process in smaller batches to avoid rate limits
            batch_size = 20
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i+batch_size]
                try:
                    tickers = yf.Tickers(' '.join(batch))
                    for symbol, ticker in tickers.tickers.items():
                        try:
                            info = ticker.info or {}
                        except Exception:
                            continue
                        
                        price = info.get('regularMarketPrice') or 0
                        change = info.get('regularMarketChangePercent') or 0
                        volume = info.get('regularMarketVolume') or info.get('volume') or 0
                        
                        # Skip if no valid data
                        if not price or price == 0:
                            continue
                            
                        avg_volume = info.get('averageDailyVolume10Day') or info.get('averageVolume') or None
                        rvol = None
                        try:
                            if avg_volume and avg_volume > 0:
                                rvol = round(volume / avg_volume, 2)
                        except Exception:
                            rvol = None

                        float_shares = info.get('floatShares') or info.get('sharesOutstanding') or None
                        market_cap = info.get('marketCap') or None

                        record = {
                            'symbol': symbol,
                            'name': info.get('shortName') or symbol,
                            'price': float(price) if isinstance(price, (int, float)) else (float(str(price).replace(',','')) if str(price) else 0),
                            'change': float(change) if isinstance(change, (int, float)) else (float(str(change).replace('%','')) if change else 0),
                            'volume': int(volume) if isinstance(volume, (int, float)) else 0,
                            'rvol': rvol,
                            'float': int(float_shares) if isinstance(float_shares, (int, float)) else None,
                            'mcap': int(market_cap) if isinstance(market_cap, (int, float)) else None
                        }
                        
                        # Only add if we have valid price and change
                        if record['price'] > 0:
                            if record['change'] >= 0:
                                gainers_list.append(record)
                            else:
                                losers_list.append(record)
                except Exception as e:
                    logger.warning(f"Error processing batch {i}: {e}")
                    continue
        except Exception as e:
            print(f"Error fetching Indian gainers/losers from yfinance: {e}")

        # Remove duplicates based on symbol
        seen_gainers = set()
        unique_gainers = []
        for item in gainers_list:
            if item['symbol'] not in seen_gainers:
                seen_gainers.add(item['symbol'])
                unique_gainers.append(item)
        gainers_list = unique_gainers
        
        seen_losers = set()
        unique_losers = []
        for item in losers_list:
            if item['symbol'] not in seen_losers:
                seen_losers.add(item['symbol'])
                unique_losers.append(item)
        losers_list = unique_losers

        # Sort and take top 50 each (or whatever we have)
        gainers_list = sorted(gainers_list, key=lambda x: x['change'], reverse=True)[:50]
        losers_list = sorted(losers_list, key=lambda x: x['change'])[:50]

        # Format fields as strings for JSON
        def fmt_currency(v):
            try:
                return f"₹{v:,.2f}"
            except Exception:
                return "--"

        def fmt_int(v):
            try:
                return f"{v:,}"
            except Exception:
                return "--"

        def format_item(it):
            return {
                'symbol': it.get('symbol'),
                'name': it.get('name'),
                'price': fmt_currency(it.get('price') or 0),
                'change': f"{it.get('change'):+.2f}%",
                'volume': fmt_int(it.get('volume') or 0),
                'rvol': str(it.get('rvol')) if it.get('rvol') is not None else '--',
                'float': fmt_int(it.get('float')) if it.get('float') else '--',
                'mcap': fmt_int(it.get('mcap')) if it.get('mcap') else '--'
            }

        gainers_out = [format_item(x) for x in gainers_list]
        losers_out = [format_item(x) for x in losers_list]

        return {'gainers': gainers_out, 'losers': losers_out}
    return jsonify(get_cached_data('gainers_losers', fetch_data))


@app.route('/api/history')
def stock_history():
    # Returns simple historical closes for a given ticker
    from flask import request
    ticker = request.args.get('ticker', 'RELIANCE.NS')
    period = request.args.get('period', '1mo')

    key = f'history::{ticker}::{period}'
    def fetch_data():
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period=period)
            dates = hist.index.strftime('%Y-%m-%d').tolist()
            prices = hist['Close'].fillna(method='ffill').tolist()
            return {'dates': dates, 'prices': prices}
        except Exception as e:
            print(f"Error fetching history for {ticker}: {e}")
            # Return a small mock series if fetch fails
            now = datetime.now()
            dates = [(now - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30, 0, -1)]
            prices = [random.uniform(2000, 2600) for _ in dates]
            return {'dates': dates, 'prices': prices}

    return jsonify(get_cached_data(key, fetch_data))

# Heatmap — aggregate Indian stocks and their sectors
@app.route('/api/heatmap')
def market_heatmap():
    def fetch_data():
        symbols = [
            'RELIANCE.NS','TCS.NS','HDFCBANK.NS','INFY.NS','ITC.NS',
            'SBIN.NS','BHARTIARTL.NS','ICICIBANK.NS','KOTAKBANK.NS','LT.NS',
            'HINDUNILVR.NS','AXISBANK.NS','ASIANPAINT.NS','MARUTI.NS','BAJFINANCE.NS',
            'WIPRO.NS','ULTRACEMCO.NS','NESTLEIND.NS','TITAN.NS','HDFCLIFE.NS'
        ]

        sector_map = {
            'RELIANCE.NS': 'Energy',
            'TCS.NS': 'Technology',
            'HDFCBANK.NS': 'Financials',
            'INFY.NS': 'Technology',
            'ITC.NS': 'Consumer',
            'SBIN.NS': 'Financials',
            'BHARTIARTL.NS': 'Communication',
            'ICICIBANK.NS': 'Financials',
            'KOTAKBANK.NS': 'Financials',
            'LT.NS': 'Industrials',
            'HINDUNILVR.NS': 'Consumer',
            'AXISBANK.NS': 'Financials',
            'ASIANPAINT.NS': 'Consumer',
            'MARUTI.NS': 'Automobile',
            'BAJFINANCE.NS': 'Financials',
            'WIPRO.NS': 'Technology',
            'ULTRACEMCO.NS': 'Materials',
            'NESTLEIND.NS': 'Consumer',
            'TITAN.NS': 'Consumer',
            'HDFCLIFE.NS': 'Financials'
        }

        stocks_data = []

        try:
            tickers = yf.Tickers(" ".join(symbols))

            for symbol in symbols:
                ticker = tickers.tickers.get(symbol)
                if not ticker:
                    continue

                try:
                    hist2 = ticker.history(period="2d")["Close"]
                    if len(hist2) == 2:
                        prev_close, last_close = hist2.iloc[0], hist2.iloc[1]
                        change = ((last_close - prev_close) / prev_close) * 100
                    else:
                        last_close = hist2.iloc[-1]
                        change = 0
                except:
                    last_close = 0
                    change = 0

                sector = sector_map.get(symbol, "Misc")

                stocks_data.append({
                    "symbol": symbol,
                    "name": symbol.replace(".NS",""),
                    "sector": sector,
                    "price": round(float(last_close), 2),
                    "change": round(float(change), 2),
                    "value": round(abs(change), 2) or 1
                })

        except Exception as e:
            print(f"Heatmap fetch failed: {e}")
            stocks_data = []

        return {"stocks": stocks_data}

    return jsonify(get_cached_data("heatmap", fetch_data))

@app.route('/api/sector_performance')
def sector_performance():
    def fetch_data():
        symbols = [
            'RELIANCE.NS','TCS.NS','HDFCBANK.NS','INFY.NS','ITC.NS',
            'SBIN.NS','BHARTIARTL.NS','ICICIBANK.NS','KOTAKBANK.NS','LT.NS',
            'HINDUNILVR.NS','AXISBANK.NS','ASIANPAINT.NS','MARUTI.NS','BAJFINANCE.NS',
            'WIPRO.NS','ULTRACEMCO.NS','NESTLEIND.NS','TITAN.NS','HDFCLIFE.NS'
        ]
        sectors_map = {}
        try:
            tickers = yf.Tickers(' '.join(symbols))
            for symbol, ticker in tickers.tickers.items():
                info = ticker.info
                sector = info.get('sector', 'Unknown')
                change = info.get('regularMarketChangePercent', 0)
                sectors_map.setdefault(sector, []).append(change)
            sectors_data = [{'name': s, 'performance': (sum(vals)/len(vals) if vals else 0)} for s, vals in sectors_map.items()]
        except Exception as e:
            print(f"Error fetching Indian sector performance: {e}")
            sectors_data = []
        return {'sectors': sectors_data}
    return jsonify(get_cached_data('sector_performance', fetch_data))

@app.route('/api/market_volatility_snapshot')
def market_volatility_snapshot():
    """
    Get current market volatility snapshot for NIFTY-50 index.
    Returns annualized volatility and risk level classification.
    """
    def fetch_data():
        try:
            logger.info("Fetching market volatility snapshot")
            
            # Fetch NIFTY-50 index data
            nifty = yf.Ticker("^NSEI")
            hist = nifty.history(period="3mo")  # Use 3 months for better volatility estimate
            
            if hist.empty or 'Close' not in hist.columns:
                logger.warning("No data available for NIFTY index")
                return {
                    'status': 'error',
                    'message': 'Unable to fetch market data',
                    'annualized_volatility': 0.0,
                    'risk_level': 'Unknown'
                }
            
            # Calculate log returns
            prices = hist['Close'].dropna()
            if len(prices) < 20:
                logger.warning("Insufficient data points for volatility calculation")
                return {
                    'status': 'error',
                    'message': 'Insufficient data',
                    'annualized_volatility': 0.0,
                    'risk_level': 'Unknown'
                }
            
            returns = np.log(prices / prices.shift(1)).dropna()
            
            # Calculate annualized volatility (assuming 252 trading days)
            daily_volatility = returns.std()
            annualized_volatility = daily_volatility * np.sqrt(252)
            
            # Determine risk level
            if annualized_volatility < 0.10:
                risk_level = 'Low'
            elif annualized_volatility < 0.20:
                risk_level = 'Moderate'
            else:
                risk_level = 'High'
            
            logger.info(f"Market volatility snapshot: {annualized_volatility:.4f} ({risk_level})")
            
            return {
                'status': 'success',
                'annualized_volatility': float(annualized_volatility),
                'risk_level': risk_level,
                'daily_volatility': float(daily_volatility),
                'period': '3mo',
                'data_points': len(returns)
            }
            
        except Exception as e:
            logger.error(f"Error in market volatility snapshot: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': f'Error calculating volatility: {str(e)}',
                'annualized_volatility': 0.0,
                'risk_level': 'Unknown'
            }
    
    return jsonify(get_cached_data('market_volatility_snapshot', fetch_data))

@app.route('/api/volatility_ranking')
def volatility_ranking_api():
    """
    Get top 5 most volatile and least volatile stocks from NIFTY-50.
    """
    from flask import request
    raw_period = request.args.get('period', '3mo')
    period = normalize_period(raw_period, default='3mo')
    
    logger.info(f"Volatility ranking request: raw_period={raw_period}, period={period}")
    
    def fetch_data():
        try:
            sector_map = get_nifty50_sector_mapping()
            nifty50_symbols = [s.lstrip('$') for s in list(sector_map.keys())]
            
            stock_volatilities = []
            
            logger.info(f"Calculating volatility for {len(nifty50_symbols)} NIFTY-50 stocks")
            
            for symbol in nifty50_symbols:
                try:
                    clean_symbol = symbol.lstrip('$')
                    stock = yf.Ticker(clean_symbol)
                    hist = stock.history(period=period)
                    
                    if not hist.empty and 'Close' in hist.columns:
                        prices = hist['Close'].dropna()
                        if len(prices) > 20:
                            returns = np.log(prices / prices.shift(1)).dropna()
                            if len(returns) > 10:
                                daily_vol = returns.std()
                                annualized_vol = daily_vol * np.sqrt(252)
                                
                                stock_volatilities.append({
                                    'symbol': clean_symbol.replace('.NS', ''),
                                    'full_symbol': clean_symbol,
                                    'volatility': float(annualized_vol),
                                    'volatility_percent': float(annualized_vol * 100),
                                    'sector': sector_map.get(clean_symbol, sector_map.get(f'${clean_symbol}', 'Unknown'))
                                })
                except Exception as e:
                    logger.debug(f"Error calculating volatility for {symbol}: {e}")
                    continue
            
            if len(stock_volatilities) < 5:
                logger.warning(f"Only {len(stock_volatilities)} stocks with valid volatility data")
                return {
                    'status': 'error',
                    'message': 'Insufficient data for ranking',
                    'most_volatile': [],
                    'least_volatile': []
                }
            
            # Sort by volatility
            sorted_stocks = sorted(stock_volatilities, key=lambda x: x['volatility'], reverse=True)
            
            most_volatile = sorted_stocks[:5]
            least_volatile = sorted_stocks[-5:][::-1]  # Reverse to show lowest first
            
            logger.info(f"Volatility ranking completed: {len(stock_volatilities)} stocks analyzed")
            
            return {
                'status': 'success',
                'period': period,
                'most_volatile': most_volatile,
                'least_volatile': least_volatile,
                'total_stocks': len(stock_volatilities)
            }
            
        except Exception as e:
            logger.error(f"Error in volatility ranking: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': f'Error calculating volatility ranking: {str(e)}',
                'most_volatile': [],
                'least_volatile': []
            }

    return jsonify(get_cached_data(f'volatility_ranking_{period}', fetch_data))

@app.route('/api/sector_rotations')
def sector_rotations():
    """
    Return time-series per sector over the past ~35 trading days.
    Attempts to fetch historical closes for each ticker and averages by sector.
    Falls back to mock data when real fetch fails.
    """
    def fetch_data():
        # mapping of sectors to representative tickers (subset)
        sector_groups = {
            'Technology': ['INFY.NS','WIPRO.NS','HCLTECH.NS','TCS.NS','TECHM.NS'],
            'Financials': ['HDFCBANK.NS','ICICIBANK.NS','KOTAKBANK.NS','AXISBANK.NS','SBIN.NS'],
            'Consumer': ['HINDUNILVR.NS','ITC.NS','BRITANNIA.NS','TITAN.NS'],
            'Energy': ['ONGC.NS','BPCL.NS','RELIANCE.NS','ADANIPORTS.NS'],
            'Industrials': ['LT.NS','JSWSTEEL.NS','TATASTEEL.NS','Bajaj-AUTO.NS'.upper()],
            'Healthcare': ['SUNPHARMA.NS','DRREDDY.NS','CIPLA.NS','DIVISLAB.NS'],
            'Materials': ['GRASIM.NS','SRF.NS','PAGEIND.NS'],
            'Utilities': ['NTPC.NS','POWERGRID.NS'],
            'Real Estate': ['DLF.NS'],
            'Communication': ['BHARTIARTL.NS']
        }

        days = 40
        try:
            # try to fetch historical prices for all unique tickers
            all_symbols = []
            for lst in sector_groups.values():
                for s in lst:
                    if s not in all_symbols:
                        all_symbols.append(s)

            # Use yf.download for multiple tickers
            import pandas as _pd
            try:
                data = yf.download(all_symbols, period=f"{days}d", progress=False, threads=False)
            except Exception:
                # fallback to single ticker downloads if multi fails
                data = None

            if data is None or data.empty:
                raise RuntimeError('yfinance download failed')

            # data can be multiindex columns (Close)
            if isinstance(data.columns, _pd.MultiIndex):
                closes = data['Close']
            elif 'Close' in data:
                closes = data['Close']
            else:
                closes = data

            # Ensure dates sorted
            closes = closes.dropna(axis=0, how='all')
            dates = closes.index.strftime('%Y-%m-%d').tolist()

            series = []
            for sector, symbols in sector_groups.items():
                # collect close series for symbols present
                vals = []
                for d in closes.index:
                    # compute average percent change vs first available day for sector
                    pass

                # compute a simple average percent change series per day
                # gather per-symbol pct change from first
                per_symbol = []
                for sym in symbols:
                    if sym in closes.columns:
                        col = closes[sym].ffill().dropna()
                        if len(col) >= 2:
                            first = col.iloc[0]
                            pct = ((col - first) / first * 100).tolist()
                            # align to dates length if shorter
                            if len(pct) < len(dates):
                                pct = [None] * (len(dates) - len(pct)) + pct
                            per_symbol.append(pct)

                if per_symbol:
                    # average across symbols (ignore None)
                    avg = []
                    for i in range(len(dates)):
                        vals_i = [p[i] for p in per_symbol if p[i] is not None]
                        if vals_i:
                            avg.append(sum(vals_i) / len(vals_i))
                        else:
                            avg.append(0)
                    series.append({'name': sector, 'values': [round(float(x), 2) if x is not None else 0 for x in avg]})

            if not series:
                raise RuntimeError('No series constructed')

            return {'dates': dates, 'series': series}
        except Exception as e:
            # fallback: create mock rotation series for each sector
            print(f"Sector rotations fetch failed: {e}")
            dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(35, 0, -1)]
            colors = ['#636efa','#ef553b','#00cc96','#ab63fa','#ffa15a','#19d3f3','#ff6692','#b6e880','#ff97ff','#fECB52']
            out = []
            i = 0
            for sector in ['Technology','Materials','Communication','Energy','Financials','Industrials','Technology (2)','Consumer Staples','Real Estate','Health Care']:
                base = random.uniform(-2,2)
                vals = []
                for d in dates:
                    base += random.uniform(-0.8,0.8)
                    vals.append(round(base + random.uniform(-0.6,0.6),2))
                out.append({'name': sector, 'values': vals, 'color': colors[i % len(colors)]})
                i += 1
            return {'dates': dates, 'series': out}

    return jsonify(get_cached_data('sector_rotations', fetch_data))

# Time Series Analysis API endpoints
def fit_arima_model(series: pd.Series, forecast_steps: int = 7) -> Dict[str, Any]:
    """
    Fit an ARIMA model to a time series and produce forecasts with confidence intervals.
    
    Args:
        series: Cleaned price or return series
        forecast_steps: Number of steps ahead to forecast (default: 7 days)
    
    Returns:
        Dictionary with model results, forecasts, and metrics
    """
    logger.info(f"Starting ARIMA model fitting for series with {len(series)} data points")
    
    if not STATSMODELS_AVAILABLE:
        logger.warning("statsmodels not available, using naive forecast fallback")
        return get_naive_forecast(series, forecast_steps)
    
    try:
        # Check for sufficient data
        if len(series) < 30:
            logger.warning(f"Insufficient data for ARIMA ({len(series)} points), using naive forecast")
            return get_naive_forecast(series, forecast_steps)
        
        # Remove any NaN or infinite values
        series_clean = series.dropna().replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(series_clean) < 30:
            logger.warning("Insufficient clean data for ARIMA, using naive forecast")
            return get_naive_forecast(series, forecast_steps)
        
        # Ensure proper datetime index with frequency
        if not isinstance(series_clean.index, pd.DatetimeIndex):
            logger.warning("Series index is not DatetimeIndex, attempting conversion")
            try:
                series_clean.index = pd.to_datetime(series_clean.index)
            except Exception as e:
                logger.error(f"Could not convert index to datetime: {e}")
                return get_naive_forecast(series, forecast_steps)
        
        # Infer or set frequency
        if series_clean.index.freq is None:
            # Try to infer frequency
            inferred_freq = pd.infer_freq(series_clean.index)
            if inferred_freq:
                logger.info(f"Inferred frequency: {inferred_freq}")
                series_clean = series_clean.asfreq(inferred_freq)
            else:
                # Default to business days for stock data
                logger.info("Could not infer frequency, using business days (B)")
                series_clean = series_clean.asfreq('B', method='ffill')
        
        # Ensure we still have enough data after frequency adjustment
        series_clean = series_clean.dropna()
        if len(series_clean) < 30:
            logger.warning("Insufficient data after frequency adjustment, using naive forecast")
            return get_naive_forecast(series, forecast_steps)
        
        # Try to determine optimal ARIMA order using auto-selection
        # Start with simple models and increase complexity
        best_aic = np.inf
        best_model = None
        best_order = (1, 1, 1)
        
        # Try common ARIMA orders
        orders_to_try = [
            (1, 1, 1), (1, 1, 0), (0, 1, 1), (2, 1, 1), (1, 1, 2),
            (2, 1, 0), (0, 1, 2), (1, 0, 1), (2, 0, 1), (1, 0, 0)
        ]
        
        logger.info("Trying different ARIMA orders to find best fit")
        
        # Suppress statsmodels ValueWarnings during model fitting
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
            warnings.filterwarnings('ignore', message='.*date index.*frequency.*', category=UserWarning)
            
            for order in orders_to_try:
                try:
                    # Suppress specific statsmodels warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', message='.*date index.*frequency.*')
                        warnings.filterwarnings('ignore', category=UserWarning)
                        
                        model = ARIMA(series_clean, order=order)
                        fitted_model = model.fit()
                        aic = fitted_model.aic
                        
                        if aic < best_aic:
                            best_aic = aic
                            best_model = fitted_model
                            best_order = order
                except Exception as e:
                    logger.debug(f"ARIMA order {order} failed: {e}")
                    continue
        
        if best_model is None:
            logger.warning("Could not fit any ARIMA model, using naive forecast")
            return get_naive_forecast(series, forecast_steps)
        
        logger.info(f"Best ARIMA model: order={best_order}, AIC={best_aic:.2f}")
        
        # Generate forecast with confidence intervals (suppress warnings)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*date index.*frequency.*')
            warnings.filterwarnings('ignore', category=UserWarning)
            
            forecast_result = best_model.get_forecast(steps=forecast_steps)
            forecast_mean = forecast_result.predicted_mean
            forecast_conf_int = forecast_result.conf_int()
        
        # Prepare forecast dates aligned with input frequency
        last_date = series_clean.index[-1]
        freq = series_clean.index.freq or pd.infer_freq(series_clean.index) or 'B'
        
        if isinstance(last_date, pd.Timestamp):
            # Use the same frequency as input series
            try:
                forecast_dates = pd.date_range(start=last_date, periods=forecast_steps + 1, freq=freq)[1:]
            except Exception:
                # Fallback to business days
                forecast_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps)
            
            # Ensure we have the right number of dates
            forecast_dates = forecast_dates[:forecast_steps]
        else:
            forecast_dates = [f"Day_{i+1}" for i in range(forecast_steps)]
        
        # Calculate in-sample metrics
        fitted_values = best_model.fittedvalues
        residuals = best_model.resid
        rmse = np.sqrt(np.mean(residuals**2))
        
        result = {
            'status': 'success',
            'model_order': best_order,
            'aic': float(best_aic),
            'bic': float(best_model.bic),
            'rmse': float(rmse),
            'forecast': {
                'dates': [d.strftime('%Y-%m-%d') if isinstance(d, pd.Timestamp) else str(d) for d in forecast_dates[:len(forecast_mean)]],
                'values': [float(v) for v in forecast_mean],
                'lower_ci': [float(v) for v in forecast_conf_int.iloc[:, 0]],
                'upper_ci': [float(v) for v in forecast_conf_int.iloc[:, 1]]
            },
            'fitted_values': {
                'dates': [d.strftime('%Y-%m-%d') if isinstance(d, pd.Timestamp) else str(d) for d in series_clean.index],
                'values': [float(v) for v in fitted_values]
            },
            'residuals': {
                'dates': [d.strftime('%Y-%m-%d') if isinstance(d, pd.Timestamp) else str(d) for d in series_clean.index],
                'values': [float(v) for v in residuals]
            }
        }
        
        logger.info(f"ARIMA model fitting completed successfully with order {best_order}")
        return result
        
    except Exception as e:
        logger.error(f"Error fitting ARIMA model: {str(e)}", exc_info=True)
        return get_naive_forecast(series, forecast_steps)

def get_naive_forecast(series: pd.Series, forecast_steps: int = 7) -> Dict[str, Any]:
    """
    Generate a naive forecast (last observed value extended) as fallback.
    """
    logger.info("Generating naive forecast fallback")
    
    series_clean = series.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(series_clean) == 0:
        last_value = 0.0
        forecast_dates = [f"Day_{i+1}" for i in range(forecast_steps)]
    else:
        last_value = float(series_clean.iloc[-1])
        last_date = series_clean.index[-1]
        
        if isinstance(last_date, pd.Timestamp):
            # Try to use the same frequency as input series
            freq = series_clean.index.freq or pd.infer_freq(series_clean.index) or 'B'
            try:
                forecast_dates = pd.date_range(start=last_date, periods=forecast_steps + 1, freq=freq)[1:]
            except Exception:
                # Fallback to business days
                forecast_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps)
            forecast_dates = forecast_dates[:forecast_steps]
        else:
            forecast_dates = [f"Day_{i+1}" for i in range(forecast_steps)]
    
    return {
        'status': 'fallback',
        'model_order': None,
        'aic': None,
        'bic': None,
        'rmse': None,
        'forecast': {
            'dates': [d.strftime('%Y-%m-%d') if isinstance(d, pd.Timestamp) else str(d) for d in forecast_dates],
            'values': [last_value] * len(forecast_dates),
            'lower_ci': [last_value * 0.95] * len(forecast_dates),
            'upper_ci': [last_value * 1.05] * len(forecast_dates)
        },
        'fitted_values': {
            'dates': [d.strftime('%Y-%m-%d') if isinstance(d, pd.Timestamp) else str(d) for d in series_clean.index] if len(series_clean) > 0 else [],
            'values': [float(v) for v in series_clean] if len(series_clean) > 0 else []
        },
        'residuals': {
            'dates': [],
            'values': []
        },
        'message': 'Using naive forecast fallback (last value extended)'
    }

@app.route('/api/time_series/arima')
def time_series_arima():
    """
    ARIMA modeling endpoint that fits an ARIMA model and produces forecasts.
    """
    from flask import request
    ticker = request.args.get('ticker', '^NSEI')
    period = request.args.get('period', '1y')
    forecast_steps = int(request.args.get('forecast_steps', 7))
    use_returns = request.args.get('use_returns', 'false').lower() == 'true'
    
    logger.info(f"ARIMA request: ticker={ticker}, period={period}, forecast_steps={forecast_steps}, use_returns={use_returns}")
    
    def fetch_data():
        try:
            # Fetch historical price data
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if hist.empty:
                logger.warning(f"No data returned for {ticker}")
                return {
                    'status': 'error',
                    'ticker': ticker,
                    'period': period,
                    'message': 'No data available for the specified ticker and period',
                    'forecast': {'dates': [], 'values': [], 'lower_ci': [], 'upper_ci': []}
                }
            
            # Extract series and ensure proper datetime index
            if use_returns:
                # Use log returns
                prices = hist['Close'].dropna()
                series = np.log(prices / prices.shift(1)).dropna()
            else:
                # Use prices
                series = hist['Close'].dropna()
            
            # Ensure series has proper datetime index
            if not isinstance(series.index, pd.DatetimeIndex):
                try:
                    series.index = pd.to_datetime(series.index)
                except Exception as e:
                    logger.warning(f"Could not convert index to datetime: {e}")
            
            # Set frequency if not already set
            if isinstance(series.index, pd.DatetimeIndex) and series.index.freq is None:
                inferred_freq = pd.infer_freq(series.index)
                if inferred_freq:
                    series = series.asfreq(inferred_freq)
                else:
                    # Default to business days for stock data
                    series = series.asfreq('B', method='ffill')
            
            if len(series) < 10:
                logger.warning(f"Insufficient data points ({len(series)})")
                return {
                    'status': 'error',
                    'ticker': ticker,
                    'period': period,
                    'message': f'Insufficient data points ({len(series)}) for ARIMA modeling',
                    'forecast': {'dates': [], 'values': [], 'lower_ci': [], 'upper_ci': []}
                }
            
            # Fit ARIMA model
            result = fit_arima_model(series, forecast_steps)
            result['ticker'] = ticker
            result['period'] = period
            result['forecast_steps'] = forecast_steps
            result['use_returns'] = use_returns
            
            return result
            
        except Exception as e:
            logger.error(f"Error in ARIMA endpoint for {ticker}: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'ticker': ticker,
                'period': period,
                'message': f'Error processing ARIMA model: {str(e)}',
                'forecast': {'dates': [], 'values': [], 'lower_ci': [], 'upper_ci': []}
            }
    
    return jsonify(get_cached_data(f'arima_{ticker}_{period}_{forecast_steps}_{use_returns}', fetch_data))

def fit_garch_model(returns: pd.Series, forecast_steps: int = 7) -> Dict[str, Any]:
    """
    Fit a GARCH model to returns and produce volatility forecasts.
    Falls back to rolling volatility if GARCH fails.
    
    Args:
        returns: Return series (log returns or simple returns)
        forecast_steps: Number of steps ahead to forecast volatility
    
    Returns:
        Dictionary with volatility estimates, forecasts, and model parameters
    """
    logger.info(f"Starting GARCH/volatility modeling for series with {len(returns)} data points")
    
    # Always compute rolling volatility as baseline
    rolling_window = min(20, len(returns) // 4) if len(returns) > 20 else len(returns)
    rolling_vol = returns.rolling(window=rolling_window).std() * np.sqrt(252)  # Annualized
    
    # Try GARCH if arch library is available
    if ARCH_AVAILABLE and len(returns) >= 50:
        try:
            logger.info("Attempting to fit GARCH(1,1) model")
            # Fit GARCH(1,1) model
            model = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='normal')
            fitted_model = model.fit(disp='off')
            
            # Extract parameters
            params = fitted_model.params
            omega = float(params.get('omega', 0))
            alpha = float(params.get('alpha[1]', 0))
            beta = float(params.get('beta[1]', 0))
            
            # Get conditional volatility
            conditional_vol = fitted_model.conditional_volatility / 100  # Convert back from percentage
            conditional_vol_annualized = conditional_vol * np.sqrt(252)
            
            # Generate volatility forecast
            try:
                forecast = fitted_model.forecast(horizon=forecast_steps, reindex=False)
                forecast_vol = forecast.variance.iloc[-1].values ** 0.5 / 100  # Convert to decimal
                forecast_vol_annualized = forecast_vol * np.sqrt(252)
            except Exception as e:
                logger.warning(f"Error generating GARCH forecast: {e}, using last volatility")
                forecast_vol_annualized = [conditional_vol_annualized.iloc[-1]] * forecast_steps
            
            # Prepare dates
            dates = returns.index.strftime('%Y-%m-%d').tolist()
            last_date = returns.index[-1]
            if isinstance(last_date, pd.Timestamp):
                forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
                forecast_dates = [d for d in forecast_dates if d.weekday() < 5][:forecast_steps]
            else:
                forecast_dates = [f"Day_{i+1}" for i in range(forecast_steps)]
            
            result = {
                'status': 'success',
                'model_type': 'GARCH(1,1)',
                'parameters': {
                    'omega': omega,
                    'alpha': alpha,
                    'beta': beta
                },
                'historical_volatility': {
                    'dates': dates,
                    'values': [float(v) if not pd.isna(v) else 0.0 for v in conditional_vol_annualized],
                    'rolling_volatility': [float(v) if not pd.isna(v) else 0.0 for v in rolling_vol]
                },
                'volatility_forecast': {
                    'dates': [str(d) for d in forecast_dates[:len(forecast_vol_annualized)]],
                    'values': [float(v) for v in forecast_vol_annualized[:forecast_steps]]
                },
                'summary': {
                    'current_volatility': float(conditional_vol_annualized.iloc[-1]) if len(conditional_vol_annualized) > 0 else 0.0,
                    'average_volatility': float(conditional_vol_annualized.mean()),
                    'max_volatility': float(conditional_vol_annualized.max()),
                    'min_volatility': float(conditional_vol_annualized.min()),
                    'forecast_avg_volatility': float(np.mean(forecast_vol_annualized)) if len(forecast_vol_annualized) > 0 else 0.0
                }
            }
            
            logger.info(f"GARCH model fitted successfully: alpha={alpha:.4f}, beta={beta:.4f}, omega={omega:.6f}")
            return result
            
        except Exception as e:
            logger.warning(f"GARCH model fitting failed: {e}, falling back to rolling volatility")
            # Fall through to rolling volatility
    
    # Fallback: Use advanced rolling volatility
    logger.info("Using rolling volatility estimation (GARCH fallback)")
    
    # Use multiple rolling windows for better volatility estimation
    vol_short = returns.rolling(window=10).std() * np.sqrt(252)
    vol_medium = returns.rolling(window=20).std() * np.sqrt(252)
    vol_long = returns.rolling(window=60).std() * np.sqrt(252) if len(returns) >= 60 else vol_medium
    
    # Use exponential weighted moving average for volatility
    vol_ewm = returns.ewm(span=20).std() * np.sqrt(252)
    
    # Combine estimates (weighted average)
    combined_vol = (vol_short.fillna(0) * 0.3 + vol_medium.fillna(0) * 0.4 + 
                    vol_long.fillna(0) * 0.2 + vol_ewm.fillna(0) * 0.1)
    
    # Simple volatility forecast (mean reversion)
    current_vol = combined_vol.iloc[-1] if len(combined_vol) > 0 else rolling_vol.iloc[-1] if len(rolling_vol) > 0 else 0.0
    avg_vol = combined_vol.mean() if len(combined_vol) > 0 else 0.0
    forecast_vol = [current_vol * (0.7 ** i) + avg_vol * (1 - 0.7 ** i) for i in range(1, forecast_steps + 1)]
    
    dates = returns.index.strftime('%Y-%m-%d').tolist()
    last_date = returns.index[-1]
    if isinstance(last_date, pd.Timestamp):
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
        forecast_dates = [d for d in forecast_dates if d.weekday() < 5][:forecast_steps]
    else:
        forecast_dates = [f"Day_{i+1}" for i in range(forecast_steps)]
    
    return {
        'status': 'fallback',
        'model_type': 'Rolling Volatility',
        'parameters': {
            'omega': None,
            'alpha': None,
            'beta': None
        },
        'historical_volatility': {
            'dates': dates,
            'values': [float(v) if not pd.isna(v) else 0.0 for v in combined_vol],
            'rolling_volatility': [float(v) if not pd.isna(v) else 0.0 for v in rolling_vol]
        },
        'volatility_forecast': {
            'dates': [str(d) for d in forecast_dates],
            'values': [float(v) for v in forecast_vol]
        },
        'summary': {
            'current_volatility': float(current_vol),
            'average_volatility': float(avg_vol),
            'max_volatility': float(combined_vol.max()) if len(combined_vol) > 0 else 0.0,
            'min_volatility': float(combined_vol.min()) if len(combined_vol) > 0 else 0.0,
            'forecast_avg_volatility': float(np.mean(forecast_vol))
        },
        'message': 'Using rolling volatility estimation (GARCH model not available or failed)'
    }

@app.route('/api/time_series/garch')
def time_series_garch():
    """
    GARCH/volatility modeling endpoint that estimates volatility and produces forecasts.
    Falls back to rolling volatility if GARCH fails.
    """
    from flask import request
    ticker = request.args.get('ticker', '^NSEI')
    period = request.args.get('period', '1y')
    forecast_steps = int(request.args.get('forecast_steps', 7))
    
    logger.info(f"GARCH/volatility request: ticker={ticker}, period={period}, forecast_steps={forecast_steps}")
    
    def fetch_data():
        try:
            # Fetch historical price data
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if hist.empty:
                logger.warning(f"No data returned for {ticker}")
                return {
                    'status': 'error',
                    'ticker': ticker,
                    'period': period,
                    'message': 'No data available for the specified ticker and period',
                    'historical_volatility': {'dates': [], 'values': [], 'rolling_volatility': []},
                    'volatility_forecast': {'dates': [], 'values': []}
                }
            
            # Compute log returns
            prices = hist['Close'].dropna()
            returns = np.log(prices / prices.shift(1)).dropna()
            
            if len(returns) < 10:
                logger.warning(f"Insufficient data points ({len(returns)})")
                return {
                    'status': 'error',
                    'ticker': ticker,
                    'period': period,
                    'message': f'Insufficient data points ({len(returns)}) for volatility modeling',
                    'historical_volatility': {'dates': [], 'values': [], 'rolling_volatility': []},
                    'volatility_forecast': {'dates': [], 'values': []}
                }
            
            # Fit GARCH/volatility model
            result = fit_garch_model(returns, forecast_steps)
            result['ticker'] = ticker
            result['period'] = period
            result['forecast_steps'] = forecast_steps
            
            return result
            
        except Exception as e:
            logger.error(f"Error in GARCH endpoint for {ticker}: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'ticker': ticker,
                'period': period,
                'message': f'Error processing volatility model: {str(e)}',
                'historical_volatility': {'dates': [], 'values': [], 'rolling_volatility': []},
                'volatility_forecast': {'dates': [], 'values': []}
            }
    
    return jsonify(get_cached_data(f'garch_{ticker}_{period}_{forecast_steps}', fetch_data))

@app.route('/api/time_series/rolling_stats')
def time_series_rolling_stats():
    """
    Fetch price data and compute log returns, rolling mean, rolling volatility, and rolling skewness.
    Returns JSON suitable for plotting.
    """
    from flask import request
    ticker = request.args.get('ticker', '^NSEI')
    period = request.args.get('period', '1y')
    window = int(request.args.get('window', 20))
    include_skewness = request.args.get('include_skewness', 'true').lower() == 'true'
    
    logger.info(f"Time series analysis requested: ticker={ticker}, period={period}, window={window}")
    
    def fetch_data():
        try:
            # Fetch historical price data
            logger.info(f"Fetching data for {ticker} over period {period}")
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if hist.empty:
                logger.warning(f"No data returned for {ticker}, using fallback")
                return get_fallback_time_series_data(ticker, period, window, include_skewness)
            
            # Extract closing prices
            prices = hist['Close'].dropna()
            
            if len(prices) < window:
                logger.warning(f"Insufficient data points ({len(prices)}) for window size {window}, using fallback")
                return get_fallback_time_series_data(ticker, period, window, include_skewness)
            
            # Compute log returns: log(P_t / P_{t-1}) = log(P_t) - log(P_{t-1})
            log_returns = np.log(prices / prices.shift(1)).dropna()
            
            # Compute rolling statistics
            rolling_mean = log_returns.rolling(window=window).mean()
            rolling_volatility = log_returns.rolling(window=window).std() * np.sqrt(252)  # Annualized volatility
            
            # Compute rolling skewness if requested
            rolling_skewness = None
            if include_skewness and len(log_returns) >= window:
                try:
                    rolling_skewness = log_returns.rolling(window=window).skew()
                except Exception as e:
                    logger.warning(f"Error computing rolling skewness: {e}")
                    rolling_skewness = None
            
            # Prepare dates for JSON serialization
            dates = prices.index.strftime('%Y-%m-%d').tolist()
            
            # Convert to lists, handling NaN values
            prices_list = prices.ffill().bfill().tolist()
            log_returns_list = log_returns.fillna(0).tolist()
            rolling_mean_list = rolling_mean.ffill().fillna(0).tolist()
            rolling_volatility_list = rolling_volatility.ffill().fillna(0).tolist()
            
            # Align dates with log returns (first date removed due to shift)
            log_returns_dates = log_returns.index.strftime('%Y-%m-%d').tolist()
            
            result = {
                'status': 'success',
                'ticker': ticker,
                'period': period,
                'window': window,
                'dates': dates,
                'prices': [float(p) for p in prices_list],
                'log_returns': {
                    'dates': log_returns_dates,
                    'values': [float(r) for r in log_returns_list]
                },
                'rolling_mean': {
                    'dates': log_returns_dates,
                    'values': [float(m) for m in rolling_mean_list]
                },
                'rolling_volatility': {
                    'dates': log_returns_dates,
                    'values': [float(v) for v in rolling_volatility_list]
                }
            }
            
            if rolling_skewness is not None:
                rolling_skewness_list = rolling_skewness.ffill().fillna(0).tolist()
                result['rolling_skewness'] = {
                    'dates': log_returns_dates,
                    'values': [float(s) for s in rolling_skewness_list]
                }
            
            # Add summary statistics
            result['summary'] = {
                'total_return': float(log_returns.sum()),
                'mean_return': float(log_returns.mean()),
                'std_return': float(log_returns.std()),
                'annualized_volatility': float(log_returns.std() * np.sqrt(252)),
                'sharpe_ratio': float(log_returns.mean() / log_returns.std() * np.sqrt(252)) if log_returns.std() > 0 else 0.0,
                'skewness': float(log_returns.skew()) if include_skewness else None,
                'kurtosis': float(log_returns.kurtosis()),
                'min_return': float(log_returns.min()),
                'max_return': float(log_returns.max())
            }
            
            logger.info(f"Successfully computed time series statistics for {ticker}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing time series data for {ticker}: {str(e)}", exc_info=True)
            return get_fallback_time_series_data(ticker, period, window, include_skewness)
    
    return jsonify(get_cached_data(f'time_series_{ticker}_{period}_{window}', fetch_data))

def get_fallback_time_series_data(ticker: str, period: str, window: int, include_skewness: bool = True) -> Dict[str, Any]:
    """
    Generate fallback/demo time series data when real data fetch fails.
    """
    logger.info(f"Generating fallback data for {ticker}")
    
    try:
        # Try to get at least some real data for dates
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if not hist.empty:
            dates = hist.index.strftime('%Y-%m-%d').tolist()
            prices = hist['Close'].tolist()
        else:
            # Generate synthetic dates
            end_date = datetime.now()
            if period == '1mo' or period == '1m':
                start_date = end_date - timedelta(days=30)
            elif period == '3mo' or period == '3m':
                start_date = end_date - timedelta(days=90)
            elif period == '6mo' or period == '6m':
                start_date = end_date - timedelta(days=180)
            elif period == '1y' or period == '1yr':
                start_date = end_date - timedelta(days=365)
            elif period == '2y' or period == '2yr':
                start_date = end_date - timedelta(days=730)
            else:
                start_date = end_date - timedelta(days=365)
            
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            dates = [d.strftime('%Y-%m-%d') for d in dates if d.weekday() < 5]  # Business days only
            prices = [1000 + random.uniform(-50, 50) for _ in dates]
    except Exception as e:
        logger.warning(f"Error generating fallback dates: {e}")
        # Last resort: generate simple date range
        dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(252, 0, -1)]
        prices = [1000 + random.uniform(-50, 50) for _ in dates]
    
    # Generate synthetic log returns
    log_returns = [random.gauss(0, 0.02) for _ in range(len(prices) - 1)]
    log_returns_dates = dates[1:]
    
    # Compute rolling statistics on synthetic data
    log_returns_series = pd.Series(log_returns, index=pd.to_datetime(log_returns_dates))
    rolling_mean = log_returns_series.rolling(window=min(window, len(log_returns_series))).mean()
    rolling_volatility = log_returns_series.rolling(window=min(window, len(log_returns_series))).std() * np.sqrt(252)
    
    result = {
        'status': 'fallback',
        'ticker': ticker,
        'period': period,
        'window': window,
        'dates': dates,
        'prices': [float(p) for p in prices],
        'log_returns': {
            'dates': log_returns_dates,
            'values': [float(r) for r in log_returns]
        },
        'rolling_mean': {
            'dates': log_returns_dates,
            'values': [float(m) if not pd.isna(m) else 0.0 for m in rolling_mean]
        },
        'rolling_volatility': {
            'dates': log_returns_dates,
            'values': [float(v) if not pd.isna(v) else 0.0 for v in rolling_volatility]
        }
    }
    
    # Add rolling skewness if requested
    if include_skewness:
        rolling_skewness = log_returns_series.rolling(window=min(window, len(log_returns_series))).skew()
        result['rolling_skewness'] = {
            'dates': log_returns_dates,
            'values': [float(s) if not pd.isna(s) else 0.0 for s in rolling_skewness]
        }
    
    result.update({
        'summary': {
            'total_return': float(sum(log_returns)),
            'mean_return': float(np.mean(log_returns)),
            'std_return': float(np.std(log_returns)),
            'annualized_volatility': float(np.std(log_returns) * np.sqrt(252)),
            'sharpe_ratio': float(np.mean(log_returns) / np.std(log_returns) * np.sqrt(252)) if np.std(log_returns) > 0 else 0.0,
            'skewness': float(pd.Series(log_returns).skew()) if include_skewness else None,
            'kurtosis': float(pd.Series(log_returns).kurtosis()),
            'min_return': float(min(log_returns)),
            'max_return': float(max(log_returns))
        },
        'message': 'Using fallback data - real data fetch failed'
    })
    
    return result

# Sector Analytics API endpoints (stubs)
@app.route('/api/sector_analytics/returns')
def sector_analytics_returns():
    from flask import request
    period = request.args.get('period', '1m')
    metric = request.args.get('metric', 'total')
    
    # Stub response - to be implemented
    return jsonify({
        'status': 'stub',
        'period': period,
        'metric': metric,
        'sectors': [],
        'message': 'Sector returns analysis endpoint - implementation pending'
    })

@app.route('/api/sector_analytics/volatility')
def sector_analytics_volatility():
    # Stub response - to be implemented
    return jsonify({
        'status': 'stub',
        'volatility_data': [],
        'stats': {
            'highest': None,
            'lowest': None,
            'average': None,
            'spread': None
        },
        'message': 'Sector volatility analysis endpoint - implementation pending'
    })

def compute_correlation_matrix(returns_df: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
    """
    Compute correlation matrix using specified method with error handling.
    
    Args:
        returns_df: DataFrame with returns (columns = stocks/sectors, index = dates)
        method: 'pearson', 'spearman', or 'kendall'
    
    Returns:
        Correlation matrix DataFrame
    """
    try:
        if method.lower() == 'pearson':
            corr_matrix = returns_df.corr(method='pearson')
        elif method.lower() == 'spearman':
            corr_matrix = returns_df.corr(method='spearman')
        elif method.lower() == 'kendall':
            corr_matrix = returns_df.corr(method='kendall')
        else:
            logger.warning(f"Unknown correlation method {method}, using pearson")
            corr_matrix = returns_df.corr(method='pearson')
        
        return corr_matrix
    except Exception as e:
        logger.error(f"Error computing correlation matrix: {e}")
        # Return identity matrix as safe default
        n = len(returns_df.columns)
        return pd.DataFrame(np.eye(n), index=returns_df.columns, columns=returns_df.columns)

def get_safe_correlation_matrix(symbols: List[str], method: str = 'pearson') -> Dict[str, Any]:
    """
    Generate a safe default correlation matrix when computation fails.
    """
    n = len(symbols)
    matrix = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    return {
        'matrix': matrix,
        'symbols': symbols,
        'method': method,
        'status': 'fallback'
    }

@app.route('/api/sector_analytics/correlation')
def sector_analytics_correlation():
    """
    Compute correlation matrix for stocks or sectors.
    Supports Pearson, Spearman, and Kendall methods.
    """
    from flask import request
    method = request.args.get('method', 'pearson')
    raw_period = request.args.get('period', '1y')
    period = normalize_period(raw_period, default='1y')
    symbols_param = request.args.get('symbols', '')  # Comma-separated list
    
    logger.info(f"Correlation request: method={method}, raw_period={raw_period}, period={period}, symbols={symbols_param}")
    
    def fetch_data():
        try:
            # Default symbols if not provided (NIFTY-50 top stocks)
            if not symbols_param:
                symbols = [
                    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ITC.NS',
                    'SBIN.NS', 'BHARTIARTL.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'LT.NS'
                ]
            else:
                symbols = [s.strip().lstrip('$') for s in symbols_param.split(',')]  # Clean symbols
            
            # Clean all symbols (remove $ prefix)
            symbols = [s.lstrip('$') for s in symbols]
            
            if len(symbols) < 2:
                logger.warning("Need at least 2 symbols for correlation analysis")
                return get_safe_correlation_matrix(symbols, method)
            
            # Fetch returns for all symbols
            returns_data = {}
            valid_symbols = []
            
            for symbol in symbols:
                try:
                    clean_symbol = symbol.lstrip('$')
                    stock = yf.Ticker(clean_symbol)
                    hist = stock.history(period=period)
                    
                    if not hist.empty and 'Close' in hist.columns:
                        prices = hist['Close'].dropna()
                        returns = np.log(prices / prices.shift(1)).dropna()
                        if len(returns) > 10:
                            returns_data[clean_symbol] = returns
                            valid_symbols.append(clean_symbol)
                except Exception as e:
                    logger.warning(f"Error fetching data for {symbol}: {e}")
                    continue
            
            if len(valid_symbols) < 2:
                logger.warning(f"Insufficient valid symbols ({len(valid_symbols)}), using fallback")
                return get_safe_correlation_matrix(symbols, method)
            
            # Align all returns to common dates
            returns_series = [returns_data[sym] for sym in valid_symbols]
            returns_df = pd.concat(returns_series, axis=1, keys=valid_symbols)
            returns_df = returns_df.dropna()
            
            if len(returns_df) < 10:
                logger.warning(f"Insufficient overlapping data points ({len(returns_df)}), using fallback")
                return get_safe_correlation_matrix(valid_symbols, method)
            
            # Compute correlation matrix
            corr_matrix = compute_correlation_matrix(returns_df, method)
            
            # Convert to list format for JSON
            matrix_list = corr_matrix.values.tolist()
            
            result = {
                'status': 'success',
                'method': method,
                'symbols': valid_symbols,
                'correlation_matrix': matrix_list,
                'matrix_df': corr_matrix.to_dict(),  # Also provide as dict for easier access
                'data_points': len(returns_df),
                'date_range': {
                    'start': returns_df.index[0].strftime('%Y-%m-%d') if isinstance(returns_df.index[0], pd.Timestamp) else str(returns_df.index[0]),
                    'end': returns_df.index[-1].strftime('%Y-%m-%d') if isinstance(returns_df.index[-1], pd.Timestamp) else str(returns_df.index[-1])
                }
            }
            
            logger.info(f"Successfully computed {method} correlation matrix for {len(valid_symbols)} symbols")
            return result
            
        except Exception as e:
            logger.error(f"Error in correlation endpoint: {str(e)}", exc_info=True)
            symbols_list = symbols_param.split(',') if symbols_param else []
            return get_safe_correlation_matrix([s.strip() for s in symbols_list] if symbols_list else [], method)
    
    return jsonify(get_cached_data(f'correlation_{method}_{period}_{symbols_param}', fetch_data))

@app.route('/api/sector_analytics/rolling_correlation')
def sector_analytics_rolling_correlation():
    """
    Compute rolling correlation between two stocks or sectors over time.
    """
    from flask import request
    symbol1 = request.args.get('symbol1', 'RELIANCE.NS')
    symbol2 = request.args.get('symbol2', 'TCS.NS')
    period = request.args.get('period', '1y')
    window = int(request.args.get('window', 20))
    method = request.args.get('method', 'pearson')
    
    logger.info(f"Rolling correlation request: {symbol1} vs {symbol2}, window={window}, method={method}")
    
    def fetch_data():
        try:
            # Fetch returns for both symbols
            stock1 = yf.Ticker(symbol1)
            stock2 = yf.Ticker(symbol2)
            
            hist1 = stock1.history(period=period)
            hist2 = stock2.history(period=period)
            
            if hist1.empty or hist2.empty:
                logger.warning(f"Insufficient data for rolling correlation")
                return {
                    'status': 'error',
                    'symbol1': symbol1,
                    'symbol2': symbol2,
                    'message': 'Insufficient data for one or both symbols',
                    'rolling_correlation': {'dates': [], 'values': []}
                }
            
            # Compute returns
            prices1 = hist1['Close'].dropna()
            prices2 = hist2['Close'].dropna()
            
            returns1 = np.log(prices1 / prices1.shift(1)).dropna()
            returns2 = np.log(prices2 / prices2.shift(1)).dropna()
            
            # Align to common dates
            returns_df = pd.DataFrame({
                symbol1: returns1,
                symbol2: returns2
            }).dropna()
            
            if len(returns_df) < window:
                logger.warning(f"Insufficient data points ({len(returns_df)}) for window {window}")
                return {
                    'status': 'error',
                    'symbol1': symbol1,
                    'symbol2': symbol2,
                    'message': f'Insufficient data points ({len(returns_df)}) for window {window}',
                    'rolling_correlation': {'dates': [], 'values': []}
                }
            
            # Compute rolling correlation
            rolling_corr = returns_df[symbol1].rolling(window=window).corr(returns_df[symbol2])
            
            # Alternative: use pandas rolling with method parameter
            if method.lower() != 'pearson':
                # For Spearman/Kendall, compute manually on rolling windows
                rolling_corr_list = []
                for i in range(window - 1, len(returns_df)):
                    window_data = returns_df.iloc[i - window + 1:i + 1]
                    if method.lower() == 'spearman':
                        corr_val = window_data[symbol1].corr(window_data[symbol2], method='spearman')
                    elif method.lower() == 'kendall':
                        corr_val = window_data[symbol1].corr(window_data[symbol2], method='kendall')
                    else:
                        corr_val = window_data[symbol1].corr(window_data[symbol2])
                    rolling_corr_list.append(corr_val if not pd.isna(corr_val) else 0.0)
                
                rolling_corr = pd.Series(rolling_corr_list, index=returns_df.index[window - 1:])
            
            dates = [d.strftime('%Y-%m-%d') if isinstance(d, pd.Timestamp) else str(d) for d in rolling_corr.index]
            values = [float(v) if not pd.isna(v) else 0.0 for v in rolling_corr]
            
            result = {
                'status': 'success',
                'symbol1': symbol1,
                'symbol2': symbol2,
                'window': window,
                'method': method,
                'rolling_correlation': {
                    'dates': dates,
                    'values': values
                },
                'summary': {
                    'mean_correlation': float(rolling_corr.mean()) if len(rolling_corr) > 0 else 0.0,
                    'std_correlation': float(rolling_corr.std()) if len(rolling_corr) > 0 else 0.0,
                    'min_correlation': float(rolling_corr.min()) if len(rolling_corr) > 0 else 0.0,
                    'max_correlation': float(rolling_corr.max()) if len(rolling_corr) > 0 else 0.0,
                    'current_correlation': float(rolling_corr.iloc[-1]) if len(rolling_corr) > 0 else 0.0
                }
            }
            
            logger.info(f"Successfully computed rolling correlation between {symbol1} and {symbol2}")
            return result
            
        except Exception as e:
            logger.error(f"Error in rolling correlation endpoint: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'symbol1': symbol1,
                'symbol2': symbol2,
                'message': f'Error computing rolling correlation: {str(e)}',
                'rolling_correlation': {'dates': [], 'values': []}
            }
    
    return jsonify(get_cached_data(f'rolling_corr_{symbol1}_{symbol2}_{window}_{method}_{period}', fetch_data))

@app.route('/api/sector_analytics/granger_causality')
def sector_analytics_granger_causality():
    """
    Perform Granger causality test between two time series.
    Example: Banking → IT sector causality.
    """
    from flask import request
    symbol1 = request.args.get('symbol1', 'HDFCBANK.NS')  # Banking
    symbol2 = request.args.get('symbol2', 'TCS.NS')  # IT
    period = request.args.get('period', '1y')
    max_lag = int(request.args.get('max_lag', 5))
    
    logger.info(f"Granger causality test: {symbol1} → {symbol2}, max_lag={max_lag}")
    
    def fetch_data():
        try:
            if not STATSMODELS_AVAILABLE:
                logger.warning("statsmodels not available for Granger causality test")
                return {
                    'status': 'error',
                    'symbol1': symbol1,
                    'symbol2': symbol2,
                    'message': 'statsmodels library not available',
                    'causality_test': {}
                }
            
            # Fetch returns for both symbols
            stock1 = yf.Ticker(symbol1)
            stock2 = yf.Ticker(symbol2)
            
            hist1 = stock1.history(period=period)
            hist2 = stock2.history(period=period)
            
            if hist1.empty or hist2.empty:
                logger.warning(f"Insufficient data for Granger causality test")
                return {
                    'status': 'error',
                    'symbol1': symbol1,
                    'symbol2': symbol2,
                    'message': 'Insufficient data for one or both symbols',
                    'causality_test': {}
                }
            
            # Compute returns
            prices1 = hist1['Close'].dropna()
            prices2 = hist2['Close'].dropna()
            
            returns1 = np.log(prices1 / prices1.shift(1)).dropna()
            returns2 = np.log(prices2 / prices2.shift(1)).dropna()
            
            # Align to common dates
            returns_df = pd.DataFrame({
                symbol1: returns1,
                symbol2: returns2
            }).dropna()
            
            if len(returns_df) < max_lag + 10:
                logger.warning(f"Insufficient data points ({len(returns_df)}) for Granger test with max_lag {max_lag}")
                return {
                    'status': 'error',
                    'symbol1': symbol1,
                    'symbol2': symbol2,
                    'message': f'Insufficient data points ({len(returns_df)}) for Granger test',
                    'causality_test': {}
                }
            
            # Prepare data for Granger causality test
            # Test: Does symbol1 Granger-cause symbol2?
            data = returns_df[[symbol2, symbol1]].values
            
            # Perform Granger causality test
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    gc_result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
                
                # Extract p-values for each lag
                p_values = {}
                f_statistics = {}
                
                for lag in range(1, max_lag + 1):
                    if lag in gc_result:
                        test_result = gc_result[lag][0]
                        # Extract F-test p-value (most common)
                        if 'ssr_ftest' in test_result:
                            p_values[lag] = float(test_result['ssr_ftest'][1])
                            f_statistics[lag] = float(test_result['ssr_ftest'][0])
                        elif 'ssr_chi2test' in test_result:
                            p_values[lag] = float(test_result['ssr_chi2test'][1])
                            f_statistics[lag] = float(test_result['ssr_chi2test'][0])
                
                # Find best lag (lowest p-value)
                if p_values:
                    best_lag = min(p_values.keys(), key=lambda k: p_values[k])
                    best_p_value = p_values[best_lag]
                    best_f_stat = f_statistics.get(best_lag, 0.0)
                    
                    # Determine causality (p < 0.05 suggests causality)
                    is_causal = best_p_value < 0.05
                    
                    result = {
                        'status': 'success',
                        'symbol1': symbol1,
                        'symbol2': symbol2,
                        'hypothesis': f'{symbol1} Granger-causes {symbol2}',
                        'is_causal': is_causal,
                        'best_lag': best_lag,
                        'best_p_value': best_p_value,
                        'best_f_statistic': best_f_stat,
                        'all_lags': {
                            'p_values': {str(k): v for k, v in p_values.items()},
                            'f_statistics': {str(k): v for k, v in f_statistics.items()}
                        },
                        'interpretation': f"{'Yes' if is_causal else 'No'}, {symbol1} {'does' if is_causal else 'does not'} Granger-cause {symbol2} (p={best_p_value:.4f} at lag {best_lag})"
                    }
                    
                    logger.info(f"Granger causality test completed: {symbol1} → {symbol2}, causal={is_causal}, p={best_p_value:.4f}")
                    return result
                else:
                    raise ValueError("No valid test results obtained")
                    
            except Exception as e:
                logger.error(f"Error performing Granger causality test: {e}")
                return {
                    'status': 'error',
                    'symbol1': symbol1,
                    'symbol2': symbol2,
                    'message': f'Error performing Granger causality test: {str(e)}',
                    'causality_test': {}
                }
            
        except Exception as e:
            logger.error(f"Error in Granger causality endpoint: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'symbol1': symbol1,
                'symbol2': symbol2,
                'message': f'Error in Granger causality test: {str(e)}',
                'causality_test': {}
            }
    
    return jsonify(get_cached_data(f'granger_{symbol1}_{symbol2}_{max_lag}_{period}', fetch_data))

def get_mock_pca_structure(symbols: List[str]) -> Dict[str, Any]:
    """
    Generate a mock PCA structure as fallback when real PCA computation fails.
    """
    n = len(symbols)
    n_components = min(10, n)
    
    # Generate mock explained variance (decreasing)
    explained_variance = [0.35, 0.20, 0.15, 0.10, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01][:n_components]
    explained_variance_ratio = [v / sum(explained_variance) if sum(explained_variance) > 0 else 0 for v in explained_variance]
    
    # Generate mock 2D projection (random but structured)
    pca_2d = [[random.uniform(-2, 2), random.uniform(-2, 2)] for _ in symbols]
    
    # Generate mock loadings
    loadings = [[random.uniform(-0.5, 0.5) for _ in range(n_components)] for _ in symbols]
    
    cumulative_variance = []
    cumsum = 0.0
    for v in explained_variance_ratio:
        cumsum += v
        cumulative_variance.append(cumsum)
    
    return {
        'status': 'fallback',
        'symbols': symbols,
        'n_components': n_components,
        'explained_variance': explained_variance,
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance_ratio': cumulative_variance,
        'pca_2d_projection': {
            'symbols': symbols,
            'x': [p[0] for p in pca_2d],
            'y': [p[1] for p in pca_2d]
        },
        'loadings': {
            'symbols': symbols,
            'components': loadings
        },
        'mds_2d_projection': {
            'symbols': symbols,
            'x': [random.uniform(-2, 2) for _ in symbols],
            'y': [random.uniform(-2, 2) for _ in symbols]
        },
        'message': 'Using mock PCA structure - real computation failed'
    }

@app.route('/api/sector_analytics/pca')
def sector_analytics_pca():
    """
    Perform PCA on NIFTY-50 log-return matrix.
    Returns principal components, explained variance, factor loadings, and 2D projection.
    Optionally includes MDS projection for comparison.
    """
    from flask import request
    raw_period = request.args.get('period', '1y')
    period = normalize_period(raw_period, default='1y')
    n_components = int(request.args.get('n_components', 10))
    include_mds = request.args.get('include_mds', 'true').lower() == 'true'
    
    logger.info(f"PCA request: raw_period={raw_period}, period={period}, n_components={n_components}, include_mds={include_mds}")
    
    def fetch_data():
        # Capture variables from outer scope (read-only, no need for nonlocal)
        _period = period
        _n_components = n_components
        _include_mds = include_mds
        
        try:
            # NIFTY-50 stock symbols
            nifty50_symbols = [
                'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ITC.NS',
                'SBIN.NS', 'BHARTIARTL.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'LT.NS',
                'HINDUNILVR.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'BAJFINANCE.NS',
                'WIPRO.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS', 'TITAN.NS', 'HDFCLIFE.NS',
                'SUNPHARMA.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'DIVISLAB.NS', 'POWERGRID.NS',
                'GAIL.NS', 'BPCL.NS', 'HEROMOTOCO.NS', 'ADANIENT.NS', 'ADANIPORTS.NS',
                'JSWSTEEL.NS', 'TATASTEEL.NS', 'GRASIM.NS', 'HCLTECH.NS', 'TECHM.NS',
                'M&M.NS', 'CIPLA.NS', 'BRITANNIA.NS', 'BAJAJ-AUTO.NS', 'SHREECEM.NS',
                'TATAMOTORS.NS', 'COALINDIA.NS', 'ONGC.NS', 'NTPC.NS', 'LUPIN.NS',
                'DLF.NS', 'SBILIFE.NS', 'ZOMATO.NS', 'PNB.NS', 'BHARATFORG.NS'
            ]
            
            # Clean symbols (remove $ prefix if any)
            nifty50_symbols = [s.lstrip('$') for s in nifty50_symbols]
            
            # Fetch log returns for all symbols
            returns_data = {}
            valid_symbols = []
            
            logger.info(f"Fetching returns data for {len(nifty50_symbols)} NIFTY-50 stocks")
            for symbol in nifty50_symbols:
                try:
                    clean_symbol = symbol.lstrip('$')
                    stock = yf.Ticker(clean_symbol)
                    hist = stock.history(period=_period)
                    
                    if not hist.empty and 'Close' in hist.columns:
                        prices = hist['Close'].dropna()
                        if len(prices) > 10:
                            returns = np.log(prices / prices.shift(1)).dropna()
                            if len(returns) > 10:
                                returns_data[clean_symbol] = returns
                                valid_symbols.append(clean_symbol)
                except Exception as e:
                    logger.debug(f"Error fetching data for {symbol}: {e}")
                    continue
            
            if len(valid_symbols) < 3:
                logger.warning(f"Insufficient valid symbols ({len(valid_symbols)}) for PCA, using fallback")
                return get_mock_pca_structure(valid_symbols if valid_symbols else nifty50_symbols[:10])
            
            # Align all returns to common dates
            logger.info(f"Aligning returns data for {len(valid_symbols)} symbols")
            returns_series = [returns_data[sym] for sym in valid_symbols]
            returns_df = pd.concat(returns_series, axis=1, keys=valid_symbols)
            returns_df = returns_df.dropna()
            
            if len(returns_df) < 10:
                logger.warning(f"Insufficient overlapping data points ({len(returns_df)}) for PCA, using fallback")
                return get_mock_pca_structure(valid_symbols)
            
            # Adjust n_components if needed
            actual_n_components = min(_n_components, len(valid_symbols), len(returns_df))
            
            logger.info(f"Computing PCA on {len(returns_df)} time points for {len(valid_symbols)} stocks")
            
            # Prepare data matrix
            # For market structure analysis: stocks as observations (rows), time as features (columns)
            # This allows us to find which stocks cluster together based on return patterns
            data_matrix = returns_df.values.T  # Transpose: Shape (stocks, time_points)
            
            # Standardize the data (important for PCA)
            if SKLEARN_AVAILABLE:
                try:
                    scaler = StandardScaler()
                    data_scaled = scaler.fit_transform(data_matrix)
                    
                    # Perform PCA - reduces stock dimension
                    pca = PCA(n_components=actual_n_components)
                    pca_result = pca.fit_transform(data_scaled)  # Shape: (stocks, n_components)
                    
                    # Get explained variance
                    explained_variance = pca.explained_variance_.tolist()
                    explained_variance_ratio = pca.explained_variance_ratio_.tolist()
                    
                    # Get components (eigenvectors) - shape: (n_components, time_points)
                    components = pca.components_.tolist()
                    
                    # Get 2D projection (first 2 principal components) - one point per stock
                    if pca_result.shape[1] >= 2:
                        pca_2d = pca_result[:, :2]  # Shape: (stocks, 2)
                    else:
                        # If only 1 component, use it for both axes
                        pca_2d = np.column_stack([pca_result[:, 0], np.zeros(len(pca_result))])
                    
                    # Factor loadings: the transformed data (pca_result) gives the loadings
                    # Each row is a stock, each column is a component
                    # pca_result[i, j] = loading of stock i on component j
                    loadings = [[float(val) for val in row] for row in pca_result]  # Shape: (stocks, n_components)
                    
                    result = {
                        'status': 'success',
                        'symbols': valid_symbols,
                        'n_components': len(components),
                        'explained_variance': [float(v) for v in explained_variance],
                        'explained_variance_ratio': [float(v) for v in explained_variance_ratio],
                        'cumulative_variance_ratio': [float(v) for v in np.cumsum(explained_variance_ratio)],
                        'pca_2d_projection': {
                            'symbols': valid_symbols,
                            'x': [float(p[0]) for p in pca_2d],
                            'y': [float(p[1]) for p in pca_2d]
                        },
                        'loadings': {
                            'symbols': valid_symbols,
                            'components': loadings
                        },
                        'components': components  # Full component matrix
                    }
                    
                    # Optionally compute MDS
                    if _include_mds and SKLEARN_AVAILABLE:
                        try:
                            logger.info("Computing MDS projection")
                            
                            # Compute correlation matrix
                            corr_matrix = returns_df.corr().values
                            # Convert to distance matrix (1 - absolute correlation)
                            distance_matrix = 1 - np.abs(corr_matrix)
                            
                            # Perform MDS
                            mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, max_iter=300)
                            mds_result = mds.fit_transform(distance_matrix)
                            
                            result['mds_2d_projection'] = {
                                'symbols': valid_symbols,
                                'x': [float(p[0]) for p in mds_result],
                                'y': [float(p[1]) for p in mds_result]
                            }
                            result['mds_stress'] = float(mds.stress_)
                            logger.info(f"MDS completed with stress: {mds.stress_:.4f}")
                        except Exception as e:
                            logger.warning(f"MDS computation failed: {e}")
                            # Generate simple MDS fallback
                            result['mds_2d_projection'] = {
                                'symbols': valid_symbols,
                                'x': [random.uniform(-2, 2) for _ in valid_symbols],
                                'y': [random.uniform(-2, 2) for _ in valid_symbols]
                            }
                    
                    logger.info(f"PCA completed successfully: {len(components)} components, {explained_variance_ratio[0]:.2%} variance explained by PC1")
                    return result
                    
                except Exception as e:
                    logger.error(f"Error computing PCA: {e}", exc_info=True)
                    return get_mock_pca_structure(valid_symbols)
            else:
                logger.warning("sklearn not available, using mock PCA structure")
                return get_mock_pca_structure(valid_symbols)
            
        except Exception as e:
            logger.error(f"Error in PCA endpoint: {str(e)}", exc_info=True)
            return get_mock_pca_structure([])
    
    return jsonify(get_cached_data(f'pca_{period}_{n_components}_{include_mds}', fetch_data))

def get_cluster_summary(cluster_data: List[Dict], cluster_id: int) -> Dict[str, Any]:
    """
    Generate a summary for a cluster based on its characteristics.
    
    Args:
        cluster_data: List of stock data in the cluster
        cluster_id: Cluster identifier
    
    Returns:
        Dictionary with cluster summary including regime type
    """
    if not cluster_data:
        return {
            'cluster_id': cluster_id,
            'count': 0,
            'regime_type': 'Unknown',
            'avg_return': 0.0,
            'avg_volatility': 0.0,
            'description': 'Empty cluster'
        }
    
    avg_return = np.mean([s['mean_return'] for s in cluster_data])
    avg_volatility = np.mean([s['volatility'] for s in cluster_data])
    
    # Determine regime type based on characteristics
    if avg_volatility > 0.03:  # High volatility threshold
        if avg_return > 0.01:
            regime_type = 'High-Volatility Growth'
            description = 'High-risk, high-return stocks with elevated volatility'
        else:
            regime_type = 'High-Volatility Defensive'
            description = 'High-volatility stocks with lower returns (defensive positioning)'
    elif avg_volatility < 0.015:  # Low volatility threshold
        if avg_return > 0.005:
            regime_type = 'Low-Volatility Growth'
            description = 'Stable, consistent growth stocks with low volatility'
        else:
            regime_type = 'Defensive'
            description = 'Low-volatility defensive stocks (utilities, consumer staples)'
    else:  # Medium volatility
        if avg_return > 0.01:
            regime_type = 'Cyclical Growth'
            description = 'Moderate-volatility cyclical stocks with good returns'
        elif avg_return < -0.005:
            regime_type = 'Cyclical Decline'
            description = 'Stocks in cyclical decline phase'
        else:
            regime_type = 'Balanced'
            description = 'Balanced stocks with moderate risk-return profile'
    
    return {
        'cluster_id': int(cluster_id),
        'count': int(len(cluster_data)),
        'regime_type': regime_type,
        'avg_return': float(avg_return),
        'avg_volatility': float(avg_volatility),
        'description': description,
        'symbols': [s['symbol'] for s in cluster_data]
    }

@app.route('/api/sector_analytics/clustering')
def sector_analytics_clustering():
    """
    Perform unsupervised clustering to detect market regimes.
    Uses K-Means on features: mean return, volatility, and PCA coordinates.
    """
    from flask import request
    algorithm = request.args.get('algorithm', 'kmeans')
    n_clusters = int(request.args.get('n_clusters', 3))
    raw_period = request.args.get('period', '1y')
    period = normalize_period(raw_period, default='1y')
    
    logger.info(f"Clustering request: algorithm={algorithm}, n_clusters={n_clusters}, raw_period={raw_period}, period={period}")
    
    def fetch_data():
        try:
            # NIFTY-50 stock symbols
            nifty50_symbols = [
                'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ITC.NS',
                'SBIN.NS', 'BHARTIARTL.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'LT.NS',
                'HINDUNILVR.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'BAJFINANCE.NS',
                'WIPRO.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS', 'TITAN.NS', 'HDFCLIFE.NS',
                'SUNPHARMA.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'DIVISLAB.NS', 'POWERGRID.NS',
                'GAIL.NS', 'BPCL.NS', 'HEROMOTOCO.NS', 'ADANIENT.NS', 'ADANIPORTS.NS',
                'JSWSTEEL.NS', 'TATASTEEL.NS', 'GRASIM.NS', 'HCLTECH.NS', 'TECHM.NS',
                'M&M.NS', 'CIPLA.NS', 'BRITANNIA.NS', 'BAJAJ-AUTO.NS', 'SHREECEM.NS',
                'TATAMOTORS.NS', 'COALINDIA.NS', 'ONGC.NS', 'NTPC.NS', 'LUPIN.NS',
                'DLF.NS', 'SBILIFE.NS', 'ZOMATO.NS', 'PNB.NS', 'BHARATFORG.NS'
            ]
            
            # Clean symbols (remove $ prefix if any)
            nifty50_symbols = [s.lstrip('$') for s in nifty50_symbols]
            
            # Fetch returns for all symbols
            returns_data = {}
            valid_symbols = []
            
            logger.info(f"Fetching returns data for {len(nifty50_symbols)} stocks")
            for symbol in nifty50_symbols:
                try:
                    clean_symbol = symbol.lstrip('$')
                    stock = yf.Ticker(clean_symbol)
                    hist = stock.history(period=period)
                    
                    if not hist.empty and 'Close' in hist.columns:
                        prices = hist['Close'].dropna()
                        if len(prices) > 10:
                            returns = np.log(prices / prices.shift(1)).dropna()
                            if len(returns) > 10:
                                returns_data[clean_symbol] = returns
                                valid_symbols.append(clean_symbol)
                except Exception as e:
                    logger.debug(f"Error fetching data for {symbol}: {e}")
                    continue
            
            if len(valid_symbols) < n_clusters:
                logger.warning(f"Insufficient valid symbols ({len(valid_symbols)}) for {n_clusters} clusters")
                return {
                    'status': 'error',
                    'message': f'Need at least {n_clusters} symbols for clustering, got {len(valid_symbols)}',
                    'algorithm': algorithm,
                    'n_clusters': n_clusters,
                    'clusters': [],
                    'labels': []
                }
            
            # Align all returns to common dates
            logger.info(f"Aligning returns data for {len(valid_symbols)} symbols")
            returns_series = [returns_data[sym] for sym in valid_symbols]
            returns_df = pd.concat(returns_series, axis=1, keys=valid_symbols)
            returns_df = returns_df.dropna()
            
            if len(returns_df) < 10:
                logger.warning(f"Insufficient overlapping data points ({len(returns_df)}) for clustering")
                return {
                    'status': 'error',
                    'message': f'Insufficient overlapping data points ({len(returns_df)}) for clustering',
                    'algorithm': algorithm,
                    'n_clusters': n_clusters,
                    'clusters': [],
                    'labels': []
                }
            
            # Compute features for each stock
            logger.info("Computing features for clustering")
            features_list = []
            stock_features = []
            
            for symbol in valid_symbols:
                returns = returns_df[symbol]
                mean_return = float(returns.mean())
                volatility = float(returns.std())
                
                stock_features.append({
                    'symbol': symbol,
                    'mean_return': mean_return,
                    'volatility': volatility
                })
                
                # Start with return and volatility
                feature_vector = [mean_return, volatility]
                features_list.append(feature_vector)
            
            # Get PCA coordinates if available
            try:
                if SKLEARN_AVAILABLE:
                    # Compute PCA on returns matrix
                    data_matrix = returns_df.values.T  # (stocks, time_points)
                    scaler = StandardScaler()
                    data_scaled = scaler.fit_transform(data_matrix)
                    
                    pca = PCA(n_components=min(2, len(valid_symbols), data_scaled.shape[1]))
                    pca_result = pca.fit_transform(data_scaled)
                    
                    # Add first 2 PCA components as features
                    for i, symbol in enumerate(valid_symbols):
                        if i < len(pca_result):
                            features_list[i].extend([float(pca_result[i, 0]), float(pca_result[i, 1])])
                            stock_features[i]['pca_x'] = float(pca_result[i, 0])
                            stock_features[i]['pca_y'] = float(pca_result[i, 1])
                        else:
                            features_list[i].extend([0.0, 0.0])
                            stock_features[i]['pca_x'] = 0.0
                            stock_features[i]['pca_y'] = 0.0
            except Exception as e:
                logger.warning(f"Error computing PCA features: {e}, using only return and volatility")
                for stock_feat in stock_features:
                    stock_feat['pca_x'] = 0.0
                    stock_feat['pca_y'] = 0.0
            
            # Convert to numpy array
            features_array = np.array(features_list)
            
            # Standardize features for clustering
            if SKLEARN_AVAILABLE:
                try:
                    from sklearn.cluster import KMeans
                    from sklearn.preprocessing import StandardScaler as FeatureScaler
                    
                    feature_scaler = FeatureScaler()
                    features_scaled = feature_scaler.fit_transform(features_array)
                    
                    # Perform K-Means clustering
                    logger.info(f"Performing K-Means clustering with {n_clusters} clusters")
                    kmeans = KMeans(n_clusters=min(n_clusters, len(valid_symbols)), random_state=42, n_init=10)
                    labels = kmeans.fit_predict(features_scaled)
                    
                    # Convert numpy types to native Python types
                    labels = [int(label) for label in labels]
                    
                    # Organize stocks by cluster
                    clusters = {}
                    for i, (symbol, label) in enumerate(zip(valid_symbols, labels)):
                        if label not in clusters:
                            clusters[label] = []
                        clusters[label].append(stock_features[i])
                    
                    # Generate cluster summaries
                    cluster_summaries = []
                    for cluster_id in sorted(clusters.keys()):
                        summary = get_cluster_summary(clusters[cluster_id], int(cluster_id))
                        cluster_summaries.append(summary)
                    
                    # Prepare PCA coordinates for visualization
                    pca_coords = []
                    for i, symbol in enumerate(valid_symbols):
                        pca_coords.append({
                            'symbol': symbol,
                            'x': float(stock_features[i].get('pca_x', 0.0)),
                            'y': float(stock_features[i].get('pca_y', 0.0)),
                            'cluster': int(labels[i]),
                            'mean_return': float(stock_features[i]['mean_return']),
                            'volatility': float(stock_features[i]['volatility'])
                        })
                    
                    result = {
                        'status': 'success',
                        'algorithm': algorithm,
                        'n_clusters': int(len(cluster_summaries)),
                        'symbols': valid_symbols,
                        'labels': labels,
                        'cluster_summaries': cluster_summaries,
                        'pca_coordinates': pca_coords,
                        'cluster_assignments': {symbol: int(label) for symbol, label in zip(valid_symbols, labels)},
                        'statistics': {
                            'total_stocks': int(len(valid_symbols)),
                            'features_used': ['mean_return', 'volatility', 'pca_x', 'pca_y'],
                            'inertia': float(kmeans.inertia_) if hasattr(kmeans, 'inertia_') else None
                        }
                    }
                    
                    logger.info(f"Clustering completed: {len(cluster_summaries)} clusters identified")
                    return result
                    
                except Exception as e:
                    logger.error(f"Error in K-Means clustering: {e}", exc_info=True)
                    # Fallback: return single cluster
                    return {
                        'status': 'fallback',
                        'algorithm': algorithm,
                        'n_clusters': 1,
                        'message': f'K-Means clustering failed: {str(e)}, returning single cluster',
                        'symbols': valid_symbols,
                        'labels': [0] * len(valid_symbols),
                        'cluster_summaries': [{
                            'cluster_id': 0,
                            'count': int(len(valid_symbols)),
                            'regime_type': 'All Stocks',
                            'avg_return': float(np.mean([s['mean_return'] for s in stock_features])),
                            'avg_volatility': float(np.mean([s['volatility'] for s in stock_features])),
                            'description': 'Single cluster fallback - clustering algorithm failed',
                            'symbols': valid_symbols
                        }],
                        'pca_coordinates': [{
                            'symbol': s['symbol'],
                            'x': float(s.get('pca_x', 0.0)),
                            'y': float(s.get('pca_y', 0.0)),
                            'cluster': 0,
                            'mean_return': float(s['mean_return']),
                            'volatility': float(s['volatility'])
                        } for s in stock_features]
                    }
            else:
                logger.warning("sklearn not available, using fallback clustering")
                return {
                    'status': 'fallback',
                    'algorithm': algorithm,
                    'n_clusters': 1,
                    'message': 'sklearn not available, returning single cluster',
                    'symbols': valid_symbols,
                    'labels': [0] * len(valid_symbols),
                    'cluster_summaries': [{
                        'cluster_id': 0,
                        'count': int(len(valid_symbols)),
                        'regime_type': 'All Stocks',
                        'avg_return': float(np.mean([s['mean_return'] for s in stock_features])),
                        'avg_volatility': float(np.mean([s['volatility'] for s in stock_features])),
                        'description': 'Single cluster fallback - sklearn not available',
                        'symbols': valid_symbols
                    }],
                    'pca_coordinates': [{
                        'symbol': s['symbol'],
                        'x': s.get('pca_x', 0.0),
                        'y': s.get('pca_y', 0.0),
                        'cluster': 0,
                        'mean_return': s['mean_return'],
                        'volatility': s['volatility']
                    } for s in stock_features]
                }
            
        except Exception as e:
            logger.error(f"Error in clustering endpoint: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': f'Error in clustering analysis: {str(e)}',
                'algorithm': algorithm,
                'n_clusters': n_clusters,
                'clusters': [],
                'labels': []
            }
    
    return jsonify(get_cached_data(f'clustering_{algorithm}_{n_clusters}_{period}', fetch_data))

def compute_marchenko_pastur_bounds(n_stocks: int, n_time_points: int, variance: float = 1.0) -> Tuple[float, float]:
    """
    Compute Marchenko-Pastur bounds for eigenvalue spectrum.
    
    Args:
        n_stocks: Number of stocks (variables)
        n_time_points: Number of time points (observations)
        variance: Variance parameter (default 1.0 for correlation matrices)
    
    Returns:
        Tuple of (lambda_min, lambda_max)
    """
    if n_time_points == 0:
        return (0.0, 1.0)
    
    Q = n_stocks / n_time_points  # Ratio of variables to observations
    
    if Q >= 1:
        # More variables than observations - adjust
        Q = 0.99
    
    lambda_min = variance * (1 - np.sqrt(Q)) ** 2
    lambda_max = variance * (1 + np.sqrt(Q)) ** 2
    
    return (lambda_min, lambda_max)

def denoise_correlation_matrix(corr_matrix: np.ndarray, eigenvalues: np.ndarray, eigenvectors: np.ndarray, 
                               significant_indices: List[int]) -> np.ndarray:
    """
    Reconstruct correlation matrix using only significant eigenvalues.
    
    Args:
        corr_matrix: Original correlation matrix
        eigenvalues: All eigenvalues
        eigenvectors: All eigenvectors (columns)
        significant_indices: Indices of significant eigenvalues to keep
    
    Returns:
        Denoised correlation matrix
    """
    # Reconstruct using only significant eigenvalues
    denoised = np.zeros_like(corr_matrix)
    
    for idx in significant_indices:
        if idx < len(eigenvalues) and idx < eigenvectors.shape[1]:
            # Outer product of eigenvector with itself, weighted by eigenvalue
            denoised += eigenvalues[idx] * np.outer(eigenvectors[:, idx], eigenvectors[:, idx])
    
    # Ensure the diagonal is 1.0 (correlation with itself)
    np.fill_diagonal(denoised, 1.0)
    
    # Ensure symmetry
    denoised = (denoised + denoised.T) / 2
    
    # Clip to valid correlation range [-1, 1]
    denoised = np.clip(denoised, -1.0, 1.0)
    
    return denoised

@app.route('/api/sector_analytics/rmt')
def sector_analytics_rmt():
    """
    Perform Random Matrix Theory (RMT) analysis for noise filtering.
    Computes eigenvalue spectrum, applies Marchenko-Pastur bounds, and constructs denoised correlation matrix.
    """
    from flask import request
    raw_period = request.args.get('period', '1y')
    period = normalize_period(raw_period, default='1y')
    symbols_param = request.args.get('symbols', '')  # Optional: comma-separated list
    
    logger.info(f"RMT request: raw_period={raw_period}, period={period}, symbols={symbols_param}")
    
    def fetch_data():
        try:
            # Get symbols (default to NIFTY-50 top stocks)
            if not symbols_param:
                symbols = [
                    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ITC.NS',
                    'SBIN.NS', 'BHARTIARTL.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'LT.NS',
                    'HINDUNILVR.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'BAJFINANCE.NS',
                    'WIPRO.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS', 'TITAN.NS', 'HDFCLIFE.NS',
                    'SUNPHARMA.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'DIVISLAB.NS', 'POWERGRID.NS',
                    'GAIL.NS', 'BPCL.NS', 'HEROMOTOCO.NS', 'ADANIENT.NS', 'ADANIPORTS.NS',
                    'JSWSTEEL.NS', 'TATASTEEL.NS', 'GRASIM.NS', 'HCLTECH.NS', 'TECHM.NS',
                    'M&M.NS', 'CIPLA.NS', 'BRITANNIA.NS', 'BAJAJ-AUTO.NS', 'SHREECEM.NS',
                    'TATAMOTORS.NS', 'COALINDIA.NS', 'ONGC.NS', 'NTPC.NS', 'LUPIN.NS'
                ]
            else:
                symbols = [s.strip().lstrip('$') for s in symbols_param.split(',')]  # Clean symbols
            
            # Clean all symbols (remove $ prefix)
            symbols = [s.lstrip('$') for s in symbols]
            
            if len(symbols) < 3:
                logger.warning("Need at least 3 symbols for RMT analysis")
                return {
                    'status': 'error',
                    'message': 'Need at least 3 symbols for RMT analysis',
                    'eigenvalues': [],
                    'raw_correlation_matrix': [],
                    'denoised_correlation_matrix': []
                }
            
            # Fetch returns for all symbols
            returns_data = {}
            valid_symbols = []
            
            logger.info(f"Fetching returns data for {len(symbols)} symbols")
            for symbol in symbols:
                try:
                    clean_symbol = symbol.lstrip('$')
                    stock = yf.Ticker(clean_symbol)
                    hist = stock.history(period=period)
                    
                    if not hist.empty and 'Close' in hist.columns:
                        prices = hist['Close'].dropna()
                        if len(prices) > 10:
                            returns = np.log(prices / prices.shift(1)).dropna()
                            if len(returns) > 10:
                                returns_data[clean_symbol] = returns
                                valid_symbols.append(clean_symbol)
                except Exception as e:
                    logger.debug(f"Error fetching data for {symbol}: {e}")
                    continue
            
            if len(valid_symbols) < 3:
                logger.warning(f"Insufficient valid symbols ({len(valid_symbols)}) for RMT")
                return {
                    'status': 'error',
                    'message': f'Insufficient valid symbols ({len(valid_symbols)}) for RMT analysis',
                    'eigenvalues': [],
                    'raw_correlation_matrix': [],
                    'denoised_correlation_matrix': []
                }
            
            # Align all returns to common dates
            logger.info(f"Aligning returns data for {len(valid_symbols)} symbols")
            returns_series = [returns_data[sym] for sym in valid_symbols]
            returns_df = pd.concat(returns_series, axis=1, keys=valid_symbols)
            returns_df = returns_df.dropna()
            
            if len(returns_df) < 10:
                logger.warning(f"Insufficient overlapping data points ({len(returns_df)}) for RMT")
                return {
                    'status': 'error',
                    'message': f'Insufficient overlapping data points ({len(returns_df)}) for RMT analysis',
                    'eigenvalues': [],
                    'raw_correlation_matrix': [],
                    'denoised_correlation_matrix': []
                }
            
            # Compute correlation matrix
            logger.info("Computing correlation matrix for RMT analysis")
            corr_matrix = returns_df.corr().values
            n_stocks = len(valid_symbols)
            n_time_points = len(returns_df)
            
            # Compute eigenvalues and eigenvectors
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
                
                # Sort in descending order
                idx = eigenvalues.argsort()[::-1]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
                
                # Compute Marchenko-Pastur bounds
                lambda_min, lambda_max = compute_marchenko_pastur_bounds(n_stocks, n_time_points, variance=1.0)
                
                # Identify significant eigenvalues (above upper bound)
                significant_indices = [i for i, ev in enumerate(eigenvalues) if ev > lambda_max]
                noise_indices = [i for i, ev in enumerate(eigenvalues) if ev <= lambda_max]
                
                # Identify market mode (largest eigenvalue)
                market_mode_idx = 0 if len(eigenvalues) > 0 else None
                market_mode_eigenvalue = float(eigenvalues[0]) if len(eigenvalues) > 0 else None
                
                # Identify sector modes (next few significant eigenvalues)
                sector_mode_indices = [i for i in significant_indices if i > 0][:5]  # Up to 5 sector modes
                sector_modes = [float(eigenvalues[i]) for i in sector_mode_indices] if sector_mode_indices else []
                
                # Construct denoised correlation matrix
                denoised_corr = denoise_correlation_matrix(corr_matrix, eigenvalues, eigenvectors, significant_indices)
                
                # Prepare eigenvalue distribution for plotting
                eigenvalue_distribution = {
                    'eigenvalues': [float(ev) for ev in eigenvalues],
                    'marchenko_pastur_min': float(lambda_min),
                    'marchenko_pastur_max': float(lambda_max),
                    'significant_indices': significant_indices,
                    'noise_indices': noise_indices
                }
                
                result = {
                    'status': 'success',
                    'symbols': valid_symbols,
                    'n_stocks': n_stocks,
                    'n_time_points': n_time_points,
                    'eigenvalue_distribution': eigenvalue_distribution,
                    'noise_threshold': float(lambda_max),
                    'significant_eigenvalues': [float(eigenvalues[i]) for i in significant_indices],
                    'noise_eigenvalues': [float(eigenvalues[i]) for i in noise_indices],
                    'market_mode': {
                        'eigenvalue': market_mode_eigenvalue,
                        'index': market_mode_idx
                    },
                    'sector_modes': sector_modes,
                    'raw_correlation_matrix': corr_matrix.tolist(),
                    'denoised_correlation_matrix': denoised_corr.tolist(),
                    'statistics': {
                        'total_eigenvalues': len(eigenvalues),
                        'significant_count': len(significant_indices),
                        'noise_count': len(noise_indices),
                        'variance_explained_by_market': float(eigenvalues[0] / n_stocks) if len(eigenvalues) > 0 else 0.0,
                        'variance_explained_by_sectors': float(sum([eigenvalues[i] for i in sector_mode_indices]) / n_stocks) if sector_mode_indices else 0.0
                    }
                }
                
                logger.info(f"RMT analysis completed: {len(significant_indices)} significant eigenvalues out of {len(eigenvalues)} total")
                return result
                
            except Exception as e:
                logger.error(f"Error computing eigenvalues: {e}", exc_info=True)
                # Fallback to simple correlation matrix
                logger.warning("Falling back to simple correlation matrix")
                return {
                    'status': 'fallback',
                    'symbols': valid_symbols,
                    'message': f'Eigenvalue computation failed: {str(e)}, using raw correlation matrix',
                    'raw_correlation_matrix': corr_matrix.tolist(),
                    'denoised_correlation_matrix': corr_matrix.tolist(),  # Same as raw in fallback
                    'eigenvalue_distribution': {
                        'eigenvalues': [],
                        'marchenko_pastur_min': None,
                        'marchenko_pastur_max': None,
                        'significant_indices': [],
                        'noise_indices': []
                    },
                    'noise_threshold': None,
                    'significant_eigenvalues': [],
                    'noise_eigenvalues': [],
                    'market_mode': None,
                    'sector_modes': []
                }
            
        except Exception as e:
            logger.error(f"Error in RMT endpoint: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': f'Error in RMT analysis: {str(e)}',
                'eigenvalues': [],
                'raw_correlation_matrix': [],
                'denoised_correlation_matrix': []
            }
    
    return jsonify(get_cached_data(f'rmt_{period}_{symbols_param}', fetch_data))

def get_nifty50_sector_mapping() -> Dict[str, str]:
    """
    Comprehensive sector mapping for NIFTY-50 stocks.
    """
    return {
        'RELIANCE.NS': 'Energy',
        'TCS.NS': 'Technology',
        'HDFCBANK.NS': 'Financials',
        'INFY.NS': 'Technology',
        'ITC.NS': 'Consumer',
        'SBIN.NS': 'Financials',
        'BHARTIARTL.NS': 'Communication',
        'ICICIBANK.NS': 'Financials',
        'KOTAKBANK.NS': 'Financials',
        'LT.NS': 'Industrials',
        'HINDUNILVR.NS': 'Consumer',
        'AXISBANK.NS': 'Financials',
        'ASIANPAINT.NS': 'Consumer',
        'MARUTI.NS': 'Automobile',
        'BAJFINANCE.NS': 'Financials',
        'WIPRO.NS': 'Technology',
        'ULTRACEMCO.NS': 'Materials',
        'NESTLEIND.NS': 'Consumer',
        'TITAN.NS': 'Consumer',
        'HDFCLIFE.NS': 'Financials',
        'SUNPHARMA.NS': 'Healthcare',
        'DRREDDY.NS': 'Healthcare',
        'EICHERMOT.NS': 'Automobile',
        'DIVISLAB.NS': 'Healthcare',
        'POWERGRID.NS': 'Utilities',
        'GAIL.NS': 'Energy',
        'BPCL.NS': 'Energy',
        'HEROMOTOCO.NS': 'Automobile',
        'ADANIENT.NS': 'Energy',
        'ADANIPORTS.NS': 'Industrials',
        'JSWSTEEL.NS': 'Materials',
        'TATASTEEL.NS': 'Materials',
        'GRASIM.NS': 'Materials',
        'HCLTECH.NS': 'Technology',
        'TECHM.NS': 'Technology',
        'M&M.NS': 'Automobile',
        'CIPLA.NS': 'Healthcare',
        'BRITANNIA.NS': 'Consumer',
        'BAJAJ-AUTO.NS': 'Automobile',
        'SHREECEM.NS': 'Materials',
        'TATAMOTORS.NS': 'Automobile',
        'COALINDIA.NS': 'Energy',
        'ONGC.NS': 'Energy',
        'NTPC.NS': 'Utilities',
        'LUPIN.NS': 'Healthcare',
        'DLF.NS': 'Real Estate',
        'SBILIFE.NS': 'Financials',
        'ZOMATO.NS': 'Consumer',
        'PNB.NS': 'Financials',
        'BHARATFORG.NS': 'Industrials'
    }

# Simple keyword → NIFTY-50 ticker mapping for news headline tagging
COMPANY_KEYWORDS: Dict[str, str] = {
    "Reliance": "RELIANCE.NS",
    "Reliance Industries": "RELIANCE.NS",
    "Infosys": "INFY.NS",
    "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Axis Bank": "AXISBANK.NS",
    "State Bank of India": "SBIN.NS",
    "ITC": "ITC.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Maruti": "MARUTI.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Larsen & Toubro": "LT.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Dr Reddy": "DRREDDY.NS",
    "Cipla": "CIPLA.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "Adani Enterprises": "ADANIENT.NS",
    # extend as needed for more NIFTY-50 names
}

def map_headline_to_sectors(title: str, sector_map: Dict[str, str]) -> List[str]:
    """
    Map a news headline to one or more sectors using COMPANY_KEYWORDS and sector_map.
    Returns a deduplicated list of sector names. Returns [] if no match.
    """
    try:
        if not title:
            return []
        
        title_lower = title.lower()
        sectors: Set[str] = set()
        
        for keyword, symbol in COMPANY_KEYWORDS.items():
            try:
                if keyword.lower() in title_lower:
                    sector = sector_map.get(symbol)
                    if sector:
                        sectors.add(sector)
            except Exception:
                continue
        
        return list(sectors)
    except Exception as e:
        logger.error(f"Error in map_headline_to_sectors: {e}", exc_info=True)
        return []

def fetch_newsapi_headlines(query: str, page_size: int = 50) -> List[Dict[str, Any]]:
    """
    Fetch news headlines from NewsAPI.
    Returns list of articles or empty list on error.
    """
    try:
        if not NEWSAPI_KEY:
            logger.warning("NEWSAPI_KEY not set, cannot fetch news")
            return []
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'sortBy': 'publishedAt',
            'language': 'en',
            'pageSize': page_size,
            'apiKey': NEWSAPI_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])
            logger.info(f"Fetched {len(articles)} articles from NewsAPI for query: {query}")
            return articles
        else:
            logger.warning(f"NewsAPI returned status {response.status_code}: {response.text[:200]}")
            return []
            
    except Exception as e:
        logger.error(f"Error fetching NewsAPI headlines: {e}", exc_info=True)
        return []

def fetch_google_news_headlines(query: str = "Nifty 50") -> List[Dict[str, Any]]:
    """
    Fetch finance news headlines using Google News RSS (Free alternative to NewsAPI).
    Returns list of dicts: {title, description, published_at, source, url}
    in the same structure expected by the sentiment pipeline.
    """
    try:
        from urllib.parse import quote_plus
        encoded_query = quote_plus(f"{query} stock market")
        rss_url = (
            f"https://news.google.com/rss/search?"
            f"q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
        )

        feed = feedparser.parse(rss_url)
        articles: List[Dict[str, Any]] = []

        from datetime import datetime

        for entry in feed.entries:
            title = entry.get("title")
            if not title:
                continue

            desc = entry.get("summary") or ""

            published_at = None
            published_struct = entry.get("published_parsed")
            if published_struct:
                try:
                    published_at = datetime(*published_struct[:6])
                except Exception:
                    published_at = None

            source_title = None
            source = getattr(entry, "source", None)
            if source and isinstance(source, dict):
                source_title = source.get("title")

            articles.append({
                "title": title,
                "description": desc,
                "published_at": published_at,
                "source": source_title,
                "url": entry.get("link")
            })

        logger.info(f"Fetched {len(articles)} Google News articles for query='{query}'")
        return articles

    except Exception as e:
        logger.error(f"Error fetching Google News headlines: {e}", exc_info=True)
        return []

def build_sector_sentiment_from_newsapi_finbert(
    articles: List[Dict[str, Any]],
    sector_map: Dict[str, str]
) -> Dict[str, Dict[str, Any]]:
    """
    Convert a list of NewsAPI articles into a sector_sentiment dict using FinBERT.
    Output format (per sector):
      {
        "dates": [YYYY-MM-DD...],
        "values": [float...],
        "current_sentiment": float,
        "avg_sentiment": float
      }
    """
    from collections import defaultdict
    from datetime import datetime, date
    
    if not articles:
        return {}
    
    sector_day_scores: Dict[str, Dict[date, List[float]]] = defaultdict(lambda: defaultdict(list))
    
    try:
        for art in articles:
            title = art.get("title") or ""
            desc = art.get("description") or ""
            text = title  # optionally: f"{title}. {desc}"
            
            if not title and not desc:
                continue
            
            score = finbert_sentiment_score(text)
            
            published_at = art.get("publishedAt")
            if isinstance(published_at, datetime):
                day = published_at.date()
            elif isinstance(published_at, str):
                try:
                    # Parse ISO format string (NewsAPI format: "2024-01-15T10:30:00Z")
                    date_str = published_at.replace('Z', '+00:00')
                    # Handle both with and without timezone
                    if '+' in date_str or date_str.endswith('+00:00'):
                        day = datetime.fromisoformat(date_str).date()
                    else:
                        # Try parsing without timezone
                        day = datetime.strptime(published_at[:10], '%Y-%m-%d').date()
                except Exception as parse_err:
                    logger.debug(f"Could not parse date '{published_at}': {parse_err}")
                    continue
            else:
                # If we cannot parse a date, skip
                continue
            
            sectors = map_headline_to_sectors(title, sector_map)
            if not sectors:
                # Optionally: treat as generic 'Market' bucket; for now, skip
                continue
            
            for sector in sectors:
                sector_day_scores[sector][day].append(score)
        
        sector_sentiment: Dict[str, Dict[str, Any]] = {}
        
        for sector, day_map in sector_day_scores.items():
            if not day_map:
                continue
            
            sorted_days = sorted(day_map.keys())
            dates: List[str] = []
            values: List[float] = []
            
            for d in sorted_days:
                scores = day_map[d]
                if not scores:
                    continue
                mean_score = float(np.mean(scores))
                dates.append(d.strftime("%Y-%m-%d"))
                values.append(mean_score)
            
            if not dates:
                continue
            
            current_sentiment = float(values[-1])
            avg_sentiment = float(np.mean(values))
            
            sector_sentiment[sector] = {
                "dates": dates,
                "values": values,
                "current_sentiment": current_sentiment,
                "avg_sentiment": avg_sentiment
            }
        
        return sector_sentiment
    except Exception as e:
                logger.error(f"Error building sector sentiment from NewsAPI + FinBERT: {e}", exc_info=True)
                return {}

def build_sector_sentiment_from_news_finbert(
    articles: List[Dict[str, Any]],
    sector_map: Dict[str, str]
) -> Dict[str, Dict[str, Any]]:
    """
    Convert a list of news articles into a sector_sentiment dict using FinBERT.
    Works with both Google News RSS and NewsAPI formats.
    Output format (per sector):
      {
        "dates": [YYYY-MM-DD...],
        "values": [float...],
        "current_sentiment": float,
        "avg_sentiment": float
      }
    """
    from collections import defaultdict
    from datetime import datetime, date
    
    if not articles:
        return {}
    
    sector_day_scores: Dict[str, Dict[date, List[float]]] = defaultdict(lambda: defaultdict(list))
    
    try:
        for art in articles:
            title = art.get("title") or ""
            desc = art.get("description") or ""
            text = title  # optionally: f"{title}. {desc}"
            
            if not title and not desc:
                continue
            
            score = finbert_sentiment_score(text)
            
            published_at = art.get("published_at") or art.get("publishedAt")
            if isinstance(published_at, datetime):
                day = published_at.date()
            elif isinstance(published_at, str):
                try:
                    # Parse ISO format string (NewsAPI format: "2024-01-15T10:30:00Z")
                    date_str = published_at.replace('Z', '+00:00')
                    # Handle both with and without timezone
                    if '+' in date_str or date_str.endswith('+00:00'):
                        day = datetime.fromisoformat(date_str).date()
                    else:
                        # Try parsing without timezone
                        day = datetime.strptime(published_at[:10], '%Y-%m-%d').date()
                except Exception:
                    # If we cannot parse a date, skip
                    continue
            else:
                # If we cannot parse a date, skip
                continue
            
            sectors = map_headline_to_sectors(title, sector_map)
            if not sectors:
                # Optionally: route to generic 'Market' bucket; for now, skip
                continue
            
            for sector in sectors:
                sector_day_scores[sector][day].append(score)
        
        sector_sentiment: Dict[str, Dict[str, Any]] = {}
        
        for sector, day_map in sector_day_scores.items():
            if not day_map:
                continue
            
            sorted_days = sorted(day_map.keys())
            dates: List[str] = []
            values: List[float] = []
            
            for d in sorted_days:
                scores = day_map[d]
                if not scores:
                    continue
                mean_score = float(np.mean(scores))
                dates.append(d.strftime("%Y-%m-%d"))
                values.append(mean_score)
            
            if not dates:
                continue
            
            current_sentiment = float(values[-1])
            avg_sentiment = float(np.mean(values))
            
            sector_sentiment[sector] = {
                "dates": dates,
                "values": values,
                "current_sentiment": current_sentiment,
                "avg_sentiment": avg_sentiment
            }
        
        return sector_sentiment
    except Exception as e:
        logger.error(f"Error building sector sentiment from news + FinBERT: {e}", exc_info=True)
        return {}

@app.route('/api/sector_analytics/overview')
def sector_analytics_overview():
    """
    Sector-level aggregation: returns, volatility, and rankings.
    """
    from flask import request
    raw_period = request.args.get('period', '1y')
    period = normalize_period(raw_period, default='1y')
    
    logger.info(f"Sector overview request: raw_period={raw_period}, period={period}")
    
    def fetch_data():
        try:
            sector_map = get_nifty50_sector_mapping()
            nifty50_symbols = [s.lstrip('$') for s in list(sector_map.keys())]  # Remove any $ prefix
            
            # Fetch returns for all symbols
            returns_data = {}
            valid_symbols = []
            
            logger.info(f"Fetching returns data for {len(nifty50_symbols)} stocks")
            for symbol in nifty50_symbols:
                try:
                    # Clean symbol (remove $ prefix if present)
                    clean_symbol = symbol.lstrip('$')
                    stock = yf.Ticker(clean_symbol)
                    hist = stock.history(period=period)
                    
                    if not hist.empty and 'Close' in hist.columns:
                        prices = hist['Close'].dropna()
                        if len(prices) > 10:
                            returns = np.log(prices / prices.shift(1)).dropna()
                            if len(returns) > 10:
                                returns_data[clean_symbol] = returns
                                valid_symbols.append(clean_symbol)
                except Exception as e:
                    logger.debug(f"Error fetching data for {symbol}: {e}")
                    continue
            
            # Group by sector
            sector_returns = {}
            sector_stocks = {}
            
            for symbol in valid_symbols:
                # Look up sector using original symbol or cleaned symbol
                sector = sector_map.get(symbol, sector_map.get(f'${symbol}', 'Unknown'))
                if sector not in sector_returns:
                    sector_returns[sector] = []
                    sector_stocks[sector] = []
                sector_returns[sector].append(returns_data[symbol])
                sector_stocks[sector].append(symbol)
            
            # Compute sector-level metrics
            sector_metrics = []
            
            for sector, returns_list in sector_returns.items():
                if not returns_list:
                    continue
                
                # Align all returns to common dates
                returns_df = pd.concat(returns_list, axis=1, keys=range(len(returns_list)))
                returns_df = returns_df.dropna()
                
                if len(returns_df) < 10:
                    logger.warning(f"Insufficient data for sector {sector}")
                    continue
                
                # Sector-level returns (equal-weighted average)
                sector_daily_returns = returns_df.mean(axis=1)
                
                # Compute metrics
                mean_return = float(sector_daily_returns.mean())
                volatility = float(sector_daily_returns.std() * np.sqrt(252))  # Annualized
                total_return = float((sector_daily_returns.sum()))
                sharpe_ratio = float(mean_return / volatility * np.sqrt(252)) if volatility > 0 else 0.0
                
                # Weekly returns (resample to weekly)
                weekly_returns = sector_daily_returns.resample('W').sum()
                
                sector_metrics.append({
                    'sector': sector,
                    'stock_count': len(sector_stocks[sector]),
                    'stocks': sector_stocks[sector],
                    'mean_return': mean_return,
                    'volatility': volatility,
                    'total_return': total_return,
                    'sharpe_ratio': sharpe_ratio,
                    'daily_returns': {
                        'dates': [d.strftime('%Y-%m-%d') if isinstance(d, pd.Timestamp) else str(d) 
                                 for d in sector_daily_returns.index],
                        'values': [float(v) for v in sector_daily_returns.values]
                    },
                    'weekly_returns': {
                        'dates': [d.strftime('%Y-%m-%d') if isinstance(d, pd.Timestamp) else str(d) 
                                 for d in weekly_returns.index],
                        'values': [float(v) for v in weekly_returns.values]
                    }
                })
            
            # Sort by total return for rankings
            sector_metrics.sort(key=lambda x: x['total_return'], reverse=True)
            
            # Add rankings
            for i, metric in enumerate(sector_metrics):
                metric['rank'] = i + 1
                metric['quartile'] = min(4, (i // max(1, len(sector_metrics) // 4)) + 1)
            
            # Top and bottom sectors
            top_sectors = sector_metrics[:3] if len(sector_metrics) >= 3 else sector_metrics
            bottom_sectors = sector_metrics[-3:] if len(sector_metrics) >= 3 else []
            
            result = {
                'status': 'success',
                'period': period,
                'sectors': sector_metrics,
                'rankings': {
                    'top_performers': [{'sector': s['sector'], 'total_return': s['total_return'], 
                                       'sharpe_ratio': s['sharpe_ratio']} for s in top_sectors],
                    'bottom_performers': [{'sector': s['sector'], 'total_return': s['total_return'], 
                                         'sharpe_ratio': s['sharpe_ratio']} for s in bottom_sectors]
                },
                'statistics': {
                    'total_sectors': len(sector_metrics),
                    'total_stocks': len(valid_symbols),
                    'missing_stocks': len(nifty50_symbols) - len(valid_symbols)
                }
            }
            
            logger.info(f"Sector overview completed: {len(sector_metrics)} sectors analyzed")
            return result
            
        except Exception as e:
            logger.error(f"Error in sector overview: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': f'Error in sector overview: {str(e)}',
                'sectors': [],
                'rankings': {'top_performers': [], 'bottom_performers': []}
            }
    
    return jsonify(get_cached_data(f'sector_overview_{period}', fetch_data))

@app.route('/api/sector_analytics/sentiment')
def sector_analytics_sentiment():
    """
    Sector-level sentiment index using Google News + FinBERT with fallback to mock data.
    """
    from flask import request
    raw_period = request.args.get('period', '1y')
    period = normalize_period(raw_period, default='1y')
    
    logger.info(f"Sector sentiment request: raw_period={raw_period}, period={period}")
    
    def fetch_data():
        try:
            sector_map = get_nifty50_sector_mapping()
            
            # ========== 1) Primary path: Google News + FinBERT ==========
            if FINBERT_AVAILABLE:
                logger.info("Using Google News + FinBERT for sector sentiment")
                
                articles = get_cached_data(
                    f'google_news_nifty_{period}',
                    lambda: fetch_google_news_headlines("Nifty 50")
                )
                
                sector_sentiment = build_sector_sentiment_from_news_finbert(
                    articles,
                    sector_map
                )
                
                if sector_sentiment:
                    logger.info(f"Google News + FinBERT produced sentiment for {len(sector_sentiment)} sectors: {list(sector_sentiment.keys())}")
                    # Merge with mock data to ensure all sectors are represented
                    all_sectors = list(set(sector_map.values()))
                    missing_sectors = set(all_sectors) - set(sector_sentiment.keys())
                    
                    if missing_sectors:
                        logger.info(f"Adding mock sentiment for {len(missing_sectors)} missing sectors: {sorted(missing_sectors)}")
                        dates = pd.date_range(end=datetime.now(), periods=252, freq='B')
                        for sector in missing_sectors:
                            base_sent = random.uniform(-0.2, 0.2)
                            values = []
                            current = base_sent
                            for _ in dates:
                                current += random.gauss(0, 0.05)
                                current = max(-1, min(1, current))
                                values.append(current)
                            
                            sector_sentiment[sector] = {
                                "dates": [d.strftime("%Y-%m-%d") for d in dates],
                                "values": [float(v) for v in values],
                                "current_sentiment": float(values[-1]),
                                "avg_sentiment": float(np.mean(values))
                            }
                    
                    return {
                        "status": "success",
                        "period": period,
                        "sector_sentiment": sector_sentiment,
                        "note": f"Sentiment derived from Google News RSS headlines using FinBERT. {len(missing_sectors) if missing_sectors else 0} sectors supplemented with mock data."
                    }
                
                logger.warning("Google News + FinBERT produced no sector sentiment, falling back to mock")
            
            # ========== 2) Fallback path: mock random-walk sentiment ==========
            logger.info("Using mock sector sentiment fallback")
            
            sectors = list(set(sector_map.values()))
            logger.info(f"Generating mock sentiment for {len(sectors)} sectors: {sorted(sectors)}")
            sentiment_data = {}
            
            dates = pd.date_range(end=datetime.now(), periods=252, freq='B')
            for sector in sectors:
                base_sent = random.uniform(-0.2, 0.2)
                values = []
                current = base_sent
                for _ in dates:
                    current += random.gauss(0, 0.05)
                    current = max(-1, min(1, current))
                    values.append(current)
                
                sentiment_data[sector] = {
                    "dates": [d.strftime("%Y-%m-%d") for d in dates],
                    "values": [float(v) for v in values],
                    "current_sentiment": float(values[-1]),
                    "avg_sentiment": float(np.mean(values))
                }
            
            logger.info(f"Mock sentiment generated for {len(sentiment_data)} sectors: {list(sentiment_data.keys())}")
            return {
                "status": "fallback",
                "period": period,
                "sector_sentiment": sentiment_data,
                "note": "Sentiment scores are mock data (fallback) because Google News/FinBERT returned no usable data."
            }
        
        except Exception as e:
            logger.error(f"Error in sector sentiment: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Error in sector sentiment: {str(e)}",
                "sector_sentiment": {}
            }
    
    return jsonify(get_cached_data(f'sector_sentiment_{period}', fetch_data))

@app.route('/api/sector_analytics/rotation')
def sector_analytics_rotation():
    """
    Compute sector rotation metrics: sectors moving between quartiles.
    """
    from flask import request
    raw_period = request.args.get('period', '1y')
    period = normalize_period(raw_period, default='1y')
    lookback_periods = int(request.args.get('lookback_periods', 4))  # Compare last N periods
    
    logger.info(f"Sector rotation request: raw_period={raw_period}, period={period}, lookback={lookback_periods}")
    
    def fetch_data():
        try:
            # Use the overview endpoint logic to get sector data
            sector_map = get_nifty50_sector_mapping()
            nifty50_symbols = [s.lstrip('$') for s in list(sector_map.keys())]  # Remove any $ prefix
            
            # Fetch returns for all symbols
            returns_data = {}
            valid_symbols = []
            
            logger.info(f"Fetching returns data for {len(nifty50_symbols)} stocks for rotation analysis")
            for symbol in nifty50_symbols:
                try:
                    clean_symbol = symbol.lstrip('$')
                    stock = yf.Ticker(clean_symbol)
                    hist = stock.history(period=period)
                    
                    if not hist.empty and 'Close' in hist.columns:
                        prices = hist['Close'].dropna()
                        if len(prices) > 10:
                            returns = np.log(prices / prices.shift(1)).dropna()
                            if len(returns) > 10:
                                returns_data[clean_symbol] = returns
                                valid_symbols.append(clean_symbol)
                except Exception as e:
                    logger.debug(f"Error fetching data for {symbol}: {e}")
                    continue
            
            # Group by sector
            sector_returns_data = {}
            for symbol in valid_symbols:
                sector = sector_map.get(symbol, sector_map.get(f'${symbol}', 'Unknown'))
                if sector not in sector_returns_data:
                    sector_returns_data[sector] = []
                sector_returns_data[sector].append(returns_data[symbol])
            
            # Compute sector performance over time periods
            # Divide time series into periods
            rotation_data = {}
            
            for sector, returns_list in sector_returns_data.items():
                if not returns_list:
                    continue
                
                returns_df = pd.concat(returns_list, axis=1)
                returns_df = returns_df.dropna()
                sector_returns = returns_df.mean(axis=1)
                
                # Divide into periods
                period_length = len(sector_returns) // lookback_periods
                period_performances = []
                
                for i in range(lookback_periods):
                    start_idx = i * period_length
                    end_idx = (i + 1) * period_length if i < lookback_periods - 1 else len(sector_returns)
                    period_ret = sector_returns.iloc[start_idx:end_idx].sum()
                    period_performances.append(float(period_ret))
                
                rotation_data[sector] = {
                    'period_performances': period_performances,
                    'current_performance': period_performances[-1],
                    'previous_performance': period_performances[-2] if len(period_performances) > 1 else 0.0,
                    'trend': 'rising' if period_performances[-1] > period_performances[0] else 'falling'
                }
            
            # Rank sectors by current performance
            sorted_sectors = sorted(rotation_data.items(), key=lambda x: x[1]['current_performance'], reverse=True)
            n_sectors = len(sorted_sectors)
            
            # Assign quartiles and identify rotations
            rising_sectors = []
            falling_sectors = []
            previous_quartiles = {}
            
            # First pass: assign current quartiles
            for i, (sector, data) in enumerate(sorted_sectors):
                quartile = min(4, (i // max(1, (n_sectors + 3) // 4)) + 1)
                data['quartile'] = quartile
                previous_quartiles[sector] = quartile
            
            # Second pass: identify rising/falling based on performance change
            for sector, data in rotation_data.items():
                if len(data['period_performances']) >= 2:
                    perf_change = data['current_performance'] - data['previous_performance']
                    if perf_change > 0.01:  # Significant positive change
                        rising_sectors.append(sector)
                    elif perf_change < -0.01:  # Significant negative change
                        falling_sectors.append(sector)
            
            result = {
                'status': 'success',
                'period': period,
                'sector_rotations': rotation_data,
                'rising_sectors': rising_sectors,
                'falling_sectors': falling_sectors,
                'insights': [
                    f"{len(rising_sectors)} sectors showing rising momentum",
                    f"{len(falling_sectors)} sectors showing declining momentum",
                    f"Top performing sector: {sorted_sectors[0][0] if sorted_sectors else 'N/A'}",
                    f"Bottom performing sector: {sorted_sectors[-1][0] if sorted_sectors else 'N/A'}"
                ]
            }
            
            logger.info(f"Sector rotation completed: {len(rotation_data)} sectors analyzed")
            return result
            
        except Exception as e:
            logger.error(f"Error in sector rotation: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': f'Error in sector rotation: {str(e)}',
                'sector_rotations': {},
                'rising_sectors': [],
                'falling_sectors': [],
                'insights': []
            }
    
    return jsonify(get_cached_data(f'sector_rotation_{period}_{lookback_periods}', fetch_data))

@app.route('/api/sector_analytics/insights')
def sector_analytics_insights():
    """
    Generate high-level insights combining all sector analytics.
    """
    from flask import request
    raw_period = request.args.get('period', '1y')
    period = normalize_period(raw_period, default='1y')
    
    logger.info(f"Sector insights request: raw_period={raw_period}, period={period}")
    
    def fetch_data():
        try:
            
            # Get sector overview data to generate dynamic insights
            sector_map = get_nifty50_sector_mapping()
            nifty50_symbols = [s.lstrip('$') for s in list(sector_map.keys())]  # Remove any $ prefix
            
            # Fetch sector overview data
            returns_data = {}
            valid_symbols = []
            
            for symbol in nifty50_symbols:
                try:
                    clean_symbol = symbol.lstrip('$')
                    stock = yf.Ticker(clean_symbol)
                    hist = stock.history(period=period)
                    
                    if not hist.empty and 'Close' in hist.columns:
                        prices = hist['Close'].dropna()
                        if len(prices) > 10:
                            returns = np.log(prices / prices.shift(1)).dropna()
                            if len(returns) > 10:
                                returns_data[symbol] = returns
                                valid_symbols.append(symbol)
                except Exception as e:
                    logger.debug(f"Error fetching data for {symbol}: {e}")
                    continue
            
            # Group by sector and compute metrics
            sector_metrics = {}
            for symbol in valid_symbols:
                sector = sector_map.get(symbol, 'Unknown')
                if sector not in sector_metrics:
                    sector_metrics[sector] = {'returns': [], 'volatilities': []}
                sector_metrics[sector]['returns'].append(returns_data[symbol])
                sector_metrics[sector]['volatilities'].append(float(returns_data[symbol].std() * np.sqrt(252)))
            
            # Generate dynamic insights
            insights = []
            
            # Find highest/lowest volatility sectors
            sector_vols = {s: np.mean(m['volatilities']) for s, m in sector_metrics.items() if m['volatilities']}
            if sector_vols:
                highest_vol_sector = max(sector_vols.items(), key=lambda x: x[1])
                lowest_vol_sector = min(sector_vols.items(), key=lambda x: x[1])
                
                if highest_vol_sector[1] > 0.3:
                    insights.append(f"{highest_vol_sector[0]} sector showing high volatility ({highest_vol_sector[1]*100:.1f}%), indicating elevated uncertainty")
                if lowest_vol_sector[1] < 0.15:
                    insights.append(f"{lowest_vol_sector[0]} sector maintaining low volatility ({lowest_vol_sector[1]*100:.1f}%), showing defensive characteristics")
            
            # Find best/worst performing sectors
            sector_performances = {}
            for sector, metrics in sector_metrics.items():
                if metrics['returns']:
                    returns_df = pd.concat(metrics['returns'], axis=1)
                    returns_df = returns_df.dropna()
                    if len(returns_df) > 0:
                        sector_returns = returns_df.mean(axis=1)
                        sector_performances[sector] = float(sector_returns.sum())
            
            if sector_performances:
                best_sector = max(sector_performances.items(), key=lambda x: x[1])
                worst_sector = min(sector_performances.items(), key=lambda x: x[1])
                
                insights.append(f"{best_sector[0]} sector leading performance with {best_sector[1]*100:.1f}% total return")
                if worst_sector[1] < 0:
                    insights.append(f"{worst_sector[0]} sector underperforming with {worst_sector[1]*100:.1f}% total return")
            
            # Correlation insights (if we have enough sectors)
            if len(sector_metrics) >= 2:
                insights.append(f"Analyzing correlations across {len(sector_metrics)} sectors to identify market structure")
            
            # Add default insights if we don't have enough data
            if len(insights) < 3:
                insights.extend([
                    "Technology and Financials sectors typically show strong correlation in market movements",
                    "Healthcare and Consumer sectors often display defensive characteristics during volatility",
                    "Energy and Materials sectors are sensitive to commodity price cycles"
                ])
            
            result = {
                'status': 'success',
                'insights': insights[:5],  # Limit to top 5 insights
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in sector insights: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': f'Error generating insights: {str(e)}',
                'insights': []
            }
    
    return jsonify(get_cached_data(f'sector_insights_{period}', fetch_data))

@app.route('/api/sector_analytics/cross_sectional')
def sector_analytics_cross_sectional():
    """
    Cross-sectional analysis: compare sectors across multiple metrics.
    """
    from flask import request
    raw_period = request.args.get('period', '1y')
    period = normalize_period(raw_period, default='1y')
    
    logger.info(f"Cross-sectional analysis request: raw_period={raw_period}, period={period}")
    
    def fetch_data():
        try:
            # Use overview endpoint logic
            sector_map = get_nifty50_sector_mapping()
            nifty50_symbols = [s.lstrip('$') for s in list(sector_map.keys())]  # Remove any $ prefix
            
            # Fetch returns
            returns_data = {}
            valid_symbols = []
            
            for symbol in nifty50_symbols:
                try:
                    clean_symbol = symbol.lstrip('$')
                    stock = yf.Ticker(clean_symbol)
                    hist = stock.history(period=period)
                    
                    if not hist.empty and 'Close' in hist.columns:
                        prices = hist['Close'].dropna()
                        if len(prices) > 10:
                            returns = np.log(prices / prices.shift(1)).dropna()
                            if len(returns) > 10:
                                returns_data[clean_symbol] = returns
                                valid_symbols.append(clean_symbol)
                except Exception as e:
                    logger.debug(f"Error fetching data for {symbol}: {e}")
                    continue
            
            # Group by sector
            sector_returns = {}
            sector_stocks = {}
            
            for symbol in valid_symbols:
                sector = sector_map.get(symbol, sector_map.get(f'${symbol}', 'Unknown'))
                if sector not in sector_returns:
                    sector_returns[sector] = []
                    sector_stocks[sector] = []
                sector_returns[sector].append(returns_data[symbol])
                sector_stocks[sector].append(symbol)
            
            # Compute cross-sectional metrics
            cross_sectional_data = []
            market_returns = None
            
            for sector, returns_list in sector_returns.items():
                if not returns_list:
                    continue
                
                returns_df = pd.concat(returns_list, axis=1, keys=range(len(returns_list)))
                returns_df = returns_df.dropna()
                
                if len(returns_df) < 10:
                    continue
                
                sector_daily_returns = returns_df.mean(axis=1)
                
                # Use NIFTY-50 as market proxy for Beta/Alpha
                if market_returns is None:
                    try:
                        nifty = yf.Ticker("^NSEI")
                        nifty_hist = nifty.history(period=period)
                        if not nifty_hist.empty:
                            nifty_prices = nifty_hist['Close'].dropna()
                            market_returns = np.log(nifty_prices / nifty_prices.shift(1)).dropna()
                            # Align dates
                            market_returns = market_returns.reindex(sector_daily_returns.index, method='nearest')
                    except Exception as e:
                        logger.warning(f"Could not fetch market returns: {e}")
                        market_returns = None
                
                # Compute metrics
                mean_return = float(sector_daily_returns.mean())
                volatility = float(sector_daily_returns.std() * np.sqrt(252))
                total_return = float(sector_daily_returns.sum())
                sharpe_ratio = float(mean_return / volatility * np.sqrt(252)) if volatility > 0 else 0.0
                
                # Beta and Alpha (if market returns available)
                beta = 0.0
                alpha = 0.0
                if market_returns is not None and len(market_returns) > 10:
                    aligned_market = market_returns.reindex(sector_daily_returns.index).dropna()
                    aligned_sector = sector_daily_returns.reindex(aligned_market.index).dropna()
                    if len(aligned_market) > 10 and len(aligned_sector) > 10:
                        market_var = float(aligned_market.var())
                        if market_var > 0:
                            beta = float(aligned_sector.cov(aligned_market) / market_var)
                            alpha = float(mean_return - beta * aligned_market.mean())
                
                cross_sectional_data.append({
                    'sector': sector,
                    'return': total_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'beta': beta,
                    'alpha': alpha,
                    'stock_count': len(sector_stocks[sector])
                })
            
            # Sort by return
            cross_sectional_data.sort(key=lambda x: x['return'], reverse=True)
            
            # Prepare scatter plot data (Return vs Volatility)
            scatter_data = {
                'sectors': [d['sector'] for d in cross_sectional_data],
                'returns': [d['return'] for d in cross_sectional_data],
                'volatilities': [d['volatility'] for d in cross_sectional_data],
                'sharpe_ratios': [d['sharpe_ratio'] for d in cross_sectional_data]
            }
            
            result = {
                'status': 'success',
                'period': period,
                'sectors': cross_sectional_data,
                'scatter_data': scatter_data
            }
            
            logger.info(f"Cross-sectional analysis completed: {len(cross_sectional_data)} sectors")
            return result
            
        except Exception as e:
            logger.error(f"Error in cross-sectional analysis: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': f'Error in cross-sectional analysis: {str(e)}',
                'sectors': [],
                'scatter_data': {}
            }
    
    return jsonify(get_cached_data(f'cross_sectional_{period}', fetch_data))

def fetch_html(url: str, timeout: int = 10) -> str:
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-IN,en;q=0.9"
            }
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}")
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to fetch HTML: {e}") from e

def _extract_json_after_marker(text: str, marker: str) -> Optional[Dict[str, Any]]:
    idx = text.find(marker)
    if idx == -1:
        return None
    # Find first '{' after marker
    brace_start = text.find("{", idx)
    if brace_start == -1:
        return None
    # Brace matching
    depth = 0
    in_str = False
    esc = False
    result_chars = []
    for ch in text[brace_start:]:
        result_chars.append(ch)
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == "\"":
                in_str = False
            continue
        else:
            if ch == "\"":
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
    json_str = "".join(result_chars)
    try:
        return json.loads(json_str)
    except Exception:
        return None

def parse_screener_rows(app_main: Dict[str, Any]) -> List[Dict[str, Any]]:
    stores = (app_main.get("context", {})
                       .get("dispatcher", {})
                       .get("stores", {}))
    rows = []
    # Yahoo Finance screener typically stores results here
    sr = stores.get("ScreenerResultsStore") or {}
    results = sr.get("results") or {}
    rows = results.get("rows") or []
    return rows if isinstance(rows, list) else []

def _to_number(val: Any) -> Optional[float]:
    # Handles dicts with 'raw', strings like '1.2B', '12,345', or numeric
    try:
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, dict) and "raw" in val:
            return _to_number(val["raw"])
        if isinstance(val, str):
            s = val.strip().upper().replace(",", "")
            mult = 1.0
            if s.endswith("T"):
                mult, s = 1e12, s[:-1]
            elif s.endswith("B"):
                mult, s = 1e9, s[:-1]
            elif s.endswith("M"):
                mult, s = 1e6, s[:-1]
            elif s.endswith("K"):
                mult, s = 1e3, s[:-1]
            return float(s) * mult
        return None
    except Exception:
        return None

def validate_record(rec: Dict[str, Any]) -> bool:
    if not rec.get("symbol") or not isinstance(rec.get("symbol"), str):
        return False
    if not rec.get("name") or not isinstance(rec.get("name"), str):
        return False
    price = _to_number(rec.get("price"))
    change_pct = _to_number(rec.get("change_percent"))
    volume = _to_number(rec.get("volume"))
    market_cap = _to_number(rec.get("market_cap")) if rec.get("market_cap") is not None else None
    if price is None or change_pct is None or volume is None:
        return False
    if price < 0 or volume < 0:
        return False
    # Update normalized values back into rec
    rec["price"] = price
    rec["change_percent"] = change_pct
    rec["volume"] = int(volume)
    rec["market_cap"] = market_cap
    return True

def fetch_gainers_data(url: str, limit: int = 50) -> List[Dict[str, Any]]:
    html = fetch_html(url)
    app_main = _extract_json_after_marker(html, "root.App.main = ")
    if not app_main:
        raise RuntimeError("Unable to locate embedded JSON (root.App.main) on page.")
    rows = parse_screener_rows(app_main)
    records: List[Dict[str, Any]] = []
    for row in rows:
        rec = {
            "symbol": row.get("symbol") or row.get("ticker") or "",
            "name": row.get("shortName") or row.get("longName") or row.get("name") or "",
            "price": row.get("regularMarketPrice") or row.get("price"),
            "change_percent": row.get("regularMarketChangePercent") or row.get("changePercent"),
            "volume": row.get("regularMarketVolume") or row.get("volume"),
            "market_cap": row.get("marketCap"),
        }
        if validate_record(rec):
            records.append(rec)
        if len(records) >= 200:  # safety cap
            break
    # Filter for Indian tickers (.NS for NSE, .BO for BSE)
    indian = [r for r in records if isinstance(r.get("symbol"), str) and (r["symbol"].endswith(".NS") or r["symbol"].endswith(".BO"))]
    final = indian[:limit] if indian else records[:limit]
    return final

def save_datasets(records: List[Dict[str, Any]], path: str = "datasets") -> None:
    # JSON Lines format; one record per line
    try:
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    except Exception as e:
        raise RuntimeError(f"Failed to write datasets: {e}") from e

def print_table(records: List[Dict[str, Any]], max_rows: int = 20) -> None:
    headers = ["Symbol", "Name", "Price", "Change %", "Volume", "Market Cap"]
    print(f"{headers[0]:<15} {headers[1]:<30} {headers[2]:>12} {headers[3]:>10} {headers[4]:>12} {headers[5]:>14}")
    print("-" * 98)
    for r in records[:max_rows]:
        mc = f"{int(r['market_cap']):,}" if isinstance(r.get("market_cap"), (int, float)) else "N/A"
        print(f"{r['symbol']:<15} {r['name'][:29]:<30} {r['price']:>12.2f} {r['change_percent']:>10.2f} {r['volume']:>12,} {mc:>14}")

def dashboard_cli(records: List[Dict[str, Any]]) -> None:
    while True:
        print("\nIndian Stocks Dashboard (CLI)")
        print("1) Top 10 by % Change")
        print("2) Top 10 by Volume")
        print("3) Search by Symbol")
        print("4) Summary Stats")
        print("5) Show First 20")
        print("6) Quit")
        choice = input("Select option: ").strip()
        if choice == "1":
            sorted_recs = sorted(records, key=lambda r: r["change_percent"], reverse=True)
            print_table(sorted_recs, 10)
        elif choice == "2":
            sorted_recs = sorted(records, key=lambda r: r["volume"], reverse=True)
            print_table(sorted_recs, 10)
        elif choice == "3":
            q = input("Enter symbol (full or partial): ").strip().upper()
            matched = [r for r in records if q in r["symbol"].upper()]
            if matched:
                print_table(matched, min(20, len(matched)))
            else:
                print("No matches found.")
        elif choice == "4":
            if not records:
                print("No data.")
                continue
            avg_change = sum(r["change_percent"] for r in records) / len(records)
            total_vol = sum(r["volume"] for r in records)
            top_mover = max(records, key=lambda r: r["change_percent"])
            worst_mover = min(records, key=lambda r: r["change_percent"])
            print(f"Count: {len(records)} | Avg Change %: {avg_change:.2f} | Total Volume: {total_vol:,}")
            print(f"Top Mover: {top_mover['symbol']} ({top_mover['change_percent']:.2f}%)")
            print(f"Worst Mover: {worst_mover['symbol']} ({worst_mover['change_percent']:.2f}%)")
        elif choice == "5":
            print_table(records, 20)
        elif choice == "6":
            print("Goodbye.")
            break
        else:
            print("Invalid option.")

def run_cli():
    url = "https://finance.yahoo.com/markets/stocks/gainers/"
    print("Fetching Yahoo Finance gainers page...")
    try:
        records = fetch_gainers_data(url, limit=50)
        if not records:
            print("No records parsed from page.")
            return
        save_datasets(records, "datasets")
        print(f"Saved {len(records)} records to 'datasets'.")
        dashboard_cli(records)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    # CLI vs Web app runner
    parser = argparse.ArgumentParser(description="Tradytics Clone + Indian Stocks CLI Dashboard")
    parser.add_argument('--cli', action='store_true', help="Run the CLI dashboard and save data to 'datasets'")
    args, _ = parser.parse_known_args()

    if args.cli:
        run_cli()
    else:
        app.run(debug=True)