# NIFTY-50 Market Dashboard

A comprehensive financial analytics dashboard for the Indian stock market, focusing on NIFTY-50 index analysis with advanced time series modeling, sector analytics, and AI-powered sentiment analysis.

## 🚀 Quick Deployment

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- Internet connection (for fetching market data)

### Deployment Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Deshboard
   ```

2. **Create and activate virtual environment**
   ```bash
   # Create virtual environment
   python3 -m venv venv
   
   # Activate virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install manually:
   ```bash
   pip install flask yfinance pandas numpy scipy scikit-learn statsmodels arch feedparser torch transformers requests
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the dashboard**
   - Open your browser and navigate to: `http://localhost:5000`
   - The dashboard will be available at the root URL

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Technologies Used](#technologies-used)
- [Features Breakdown](#features-breakdown)

## ✨ Features

### 1. **Market Overview Dashboard**
- Real-time NIFTY-50 index tracking
- Market sentiment analysis (AI-powered)
- India VIX monitoring
- Market leaders and gainers/losers
- Interactive price charts with Plotly.js

### 2. **Time Series Analysis**
- **ARIMA Forecasting**: AutoRegressive Integrated Moving Average models for price prediction
- **GARCH Volatility Modeling**: Generalized Autoregressive Conditional Heteroskedasticity for volatility forecasting
- **Rolling Statistics**: Mean, volatility, and skewness calculations over configurable windows
- Interactive charts with confidence intervals

### 3. **Sector Analytics**
- **Sector Performance**: Returns, volatility, and rankings across 11 sectors
- **Correlation Analysis**: Pearson, Spearman, and Kendall correlation matrices
- **Principal Component Analysis (PCA)**: Dimensionality reduction to identify market structure
- **Random Matrix Theory (RMT)**: Noise filtering for correlation matrices
- **K-Means Clustering**: Market regime detection based on stock features
- **Sector Sentiment**: AI-powered sentiment analysis using Google News + FinBERT
- **Sector Rotation**: Identification of rising/falling sectors
- **Cross-Sectional Analysis**: Sharpe Ratio, Beta, Alpha calculations

### 4. **AI-Powered Sentiment Analysis**
- Real-time news aggregation from Google News RSS
- FinBERT-based sentiment scoring (financial domain-specific BERT model)
- Sector-wise sentiment mapping
- Market-wide sentiment aggregation

## 📁 Project Structure

```
Deshboard/
├── app.py                          # Main Flask application
├── templates/                      # HTML templates
│   ├── index.html                  # Main dashboard
│   ├── time_series_analysis.html   # Time series analysis page
│   ├── sector_analytics.html       # Sector analytics page
│   └── stock_predictor.html        # Stock prediction page
├── static/
│   ├── css/                        # Stylesheets
│   │   ├── style.css
│   │   └── predictor_style.css
│   └── js/                         # JavaScript files
│       ├── main.js                 # Main dashboard logic
│       ├── time_series.js          # Time series analysis
│       ├── sector_analytics.js     # Sector analytics
│       └── predictor.js            # Stock prediction
├── venv/                           # Virtual environment (not in repo)
└── README.md                       # This file
```

## 🔧 Installation

### Step-by-Step Setup

1. **Ensure Python 3.9+ is installed**
   ```bash
   python3 --version
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   ```

3. **Activate virtual environment**
   ```bash
   # macOS/Linux
   source venv/bin/activate
   
   # Windows
   venv\Scripts\activate
   ```

4. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```
   
   This will install all required dependencies:
   - Flask (web framework)
   - yfinance (market data)
   - pandas, numpy, scipy (data processing)
   - scikit-learn (machine learning)
   - statsmodels (statistical models)
   - arch (GARCH models)
   - feedparser (RSS feeds)
   - torch, transformers (FinBERT)
   - requests (HTTP requests)

5. **Verify installation**
   ```bash
   python -c "import flask, yfinance, pandas, numpy, sklearn, statsmodels, arch, feedparser, torch, transformers; print('All packages installed successfully!')"
   ```

## ⚙️ Configuration

### Environment Variables (Optional)

The application works out of the box without any configuration. However, you can set:

- `NEWSAPI_KEY`: For NewsAPI integration (currently using Google News RSS, so not required)
- `FLASK_ENV`: Set to `development` or `production`

### Cache Configuration

The application uses in-memory caching with a 5-minute TTL to reduce API calls. This is configured in `app.py`:

```python
cache_timeout = 300  # 5 minutes
```

## 🎯 Usage

### Starting the Application

1. **Activate virtual environment**
   ```bash
   source venv/bin/activate  # macOS/Linux
   # or
   venv\Scripts\activate     # Windows
   ```

2. **Run the Flask app**
   ```bash
   python app.py
   ```

3. **Access the dashboard**
   - Main Dashboard: `http://localhost:5000`
   - Time Series Analysis: `http://localhost:5000/time_series_analysis`
   - Sector Analytics: `http://localhost:5000/sector_analytics`

### Using the Dashboard

1. **Market Overview**
   - View real-time NIFTY-50 data
   - Check market sentiment (Bullish/Bearish/Neutral)
   - Monitor India VIX levels
   - Browse market leaders and gainers

2. **Time Series Analysis**
   - Select a stock ticker (e.g., `RELIANCE.NS`)
   - Choose time period (1mo, 3mo, 6mo, 1y)
   - View ARIMA forecasts with confidence intervals
   - Analyze GARCH volatility models
   - Explore rolling statistics

3. **Sector Analytics**
   - View sector performance rankings
   - Analyze correlation matrices
   - Explore PCA projections
   - Review RMT-denoised correlations
   - Check sector sentiment over time
   - Monitor sector rotation patterns

## 🔌 API Endpoints

### Market Data
- `GET /api/market_summary` - Market overview (NIFTY, VIX, sentiment)
- `GET /api/gainers_losers` - Top gainers and losers
- `GET /api/heatmap` - Stock heatmap data

### Time Series Analysis
- `GET /api/time_series/rolling_stats?ticker=RELIANCE.NS&period=1y&window=20` - Rolling statistics
- `GET /api/time_series/arima?ticker=RELIANCE.NS&period=1y&forecast_steps=7` - ARIMA forecast
- `GET /api/time_series/garch?ticker=RELIANCE.NS&period=1y&forecast_steps=7` - GARCH volatility

### Sector Analytics
- `GET /api/sector_analytics/overview?period=1y` - Sector performance overview
- `GET /api/sector_analytics/correlation?method=pearson&period=1y` - Correlation matrix
- `GET /api/sector_analytics/pca?period=1y&n_components=10` - PCA analysis
- `GET /api/sector_analytics/rmt?period=1y` - Random Matrix Theory analysis
- `GET /api/sector_analytics/clustering?algorithm=kmeans&n_clusters=3&period=1y` - Market regime clustering
- `GET /api/sector_analytics/sentiment?period=1y` - Sector sentiment analysis
- `GET /api/sector_analytics/rotation?period=1y&lookback_periods=4` - Sector rotation
- `GET /api/sector_analytics/cross_sectional?period=1y` - Cross-sectional metrics
- `GET /api/sector_analytics/insights?period=1y` - High-level insights

### Query Parameters
- `period`: Time period (`1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max`)
- `ticker`: Stock symbol (e.g., `RELIANCE.NS`, `TCS.NS`)
- `window`: Rolling window size (default: 20)
- `method`: Correlation method (`pearson`, `spearman`, `kendall`)

## 🛠️ Technologies Used

### Backend
- **Flask**: Web framework
- **yfinance**: Market data fetching
- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **scipy**: Scientific computing
- **scikit-learn**: Machine learning (PCA, K-Means)
- **statsmodels**: Statistical models (ARIMA, Granger Causality)
- **arch**: GARCH volatility modeling
- **feedparser**: RSS feed parsing (Google News)
- **PyTorch**: Deep learning framework
- **transformers**: Hugging Face transformers (FinBERT)

### Frontend
- **Plotly.js**: Interactive charts and visualizations
- **Vanilla JavaScript**: No framework dependencies
- **HTML5/CSS3**: Modern web standards

### AI/ML
- **FinBERT**: Financial sentiment analysis model (ProsusAI/finbert)
- **ARIMA**: Time series forecasting
- **GARCH**: Volatility modeling
- **PCA**: Dimensionality reduction
- **K-Means**: Unsupervised clustering
- **Random Matrix Theory**: Noise filtering

## 📊 Features Breakdown

### Market Sentiment Analysis
- **Source**: Google News RSS feeds
- **Model**: FinBERT (financial domain-specific BERT)
- **Output**: Sector-wise and market-wide sentiment scores (-1 to +1)
- **Update Frequency**: Cached for 5 minutes

### Time Series Models
- **ARIMA**: Automatically selects optimal (p,d,q) parameters
- **GARCH**: Models volatility clustering and heteroskedasticity
- **Forecasting**: 7-day ahead predictions with confidence intervals
- **Fallbacks**: Naive forecasts if model fitting fails

### Sector Analytics
- **11 Sectors**: Automobile, Communication, Consumer, Energy, Financials, Healthcare, Industrials, Materials, Real Estate, Technology, Utilities
- **50 Stocks**: Complete NIFTY-50 coverage
- **Real-time Data**: Fetched from Yahoo Finance via yfinance

### Data Sources
- **Yahoo Finance**: Stock prices, market data (via yfinance)
- **Google News RSS**: Financial news headlines
- **FinBERT**: Pre-trained model for sentiment analysis

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure virtual environment is activated
   source venv/bin/activate
   # Reinstall packages
   pip install --upgrade -r requirements.txt
   ```

2. **Port Already in Use**
   ```bash
   # Change port in app.py
   app.run(host='0.0.0.0', port=5001)  # Use different port
   ```

3. **FinBERT Model Download Issues**
   - First run downloads ~500MB model
   - Ensure stable internet connection
   - Model is cached after first download

4. **Data Fetching Errors**
   - Check internet connection
   - Yahoo Finance may have rate limits
   - Cache reduces API calls (5-minute TTL)

5. **Chart Not Displaying**
   - Check browser console for errors
   - Ensure Plotly.js CDN is accessible
   - Verify JavaScript is enabled

## 📝 Notes

- **Virtual Environment**: The `venv/` directory is not included in the repository. Users must create their own virtual environment.
- **Caching**: Data is cached for 5 minutes to reduce API calls and improve performance.
- **Period Normalization**: Invalid periods (e.g., `1m`) are automatically converted to valid ones (`1mo`).
- **Error Handling**: All endpoints include robust error handling with fallback mechanisms.
- **Logging**: Application logs are output to console with INFO level by default.

## 🔒 Security Notes

- No authentication is implemented (add if deploying publicly)
- API endpoints are open (consider rate limiting for production)
- Environment variables should be used for sensitive keys
- CORS is not configured (add if needed for cross-origin requests)

## 📄 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code follows existing style
- Error handling is robust
- Logging is informative
- Fallbacks are implemented for external API calls

## 📧 Support

For issues or questions:
1. Check the troubleshooting section
2. Review application logs
3. Check browser console for frontend errors
4. Verify all dependencies are installed correctly

---

**Built with ❤️ for the Indian Stock Market**
