// Time Series Analysis JavaScript - Stub implementation

document.addEventListener('DOMContentLoaded', function() {
    // Initialize event listeners
    const analyzeBtn = document.getElementById('ts-analyze-btn');
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', function() {
            const ticker = document.getElementById('ts-ticker-select').value;
            const period = document.getElementById('ts-period-select').value;
            analyzeTimeSeries(ticker, period);
        });
    }

    // Rolling stats controls
    const rollingWindow = document.getElementById('rolling-window');
    const rollingMetric = document.getElementById('rolling-metric');
    if (rollingWindow && rollingMetric) {
        rollingWindow.addEventListener('change', updateRollingStats);
        rollingMetric.addEventListener('change', updateRollingStats);
    }
    
    // Load initial data on page load
    setTimeout(() => {
        const ticker = document.getElementById('ts-ticker-select')?.value || '^NSEI';
        const period = document.getElementById('ts-period-select')?.value || '6mo';
        analyzeTimeSeries(ticker, period);
    }, 100);
    
    // Handle window resize for charts
    let resizeTimer;
    window.addEventListener('resize', function() {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(function() {
            const charts = ['arima-chart', 'garch-chart', 'rolling-stats-chart'];
            charts.forEach(chartId => {
                const chartDiv = document.getElementById(chartId);
                if (chartDiv && chartDiv.data) {
                    Plotly.Plots.resize(chartDiv);
                }
            });
        }, 250);
    });
});

function analyzeTimeSeries(ticker, period) {
    console.log('Analyzing time series for:', ticker, period);
    
    // Fetch ARIMA data
    fetchARIMA(ticker, period);
    
    // Fetch GARCH data
    fetchGARCH(ticker, period);
    
    // Fetch rolling stats
    updateRollingStats();
}

function fetchARIMA(ticker, period) {
    fetch(`/api/time_series/arima?ticker=${ticker}&period=${period}`)
        .then(response => response.json())
        .then(data => {
            console.log('ARIMA data:', data);
            if (data.status === 'success' || data.status === 'fallback') {
                renderARIMAChart(data);
                updateARIMAStats(data);
            } else {
                console.error('ARIMA data error:', data.message);
                showError('arima-chart', 'Error loading ARIMA data');
            }
        })
        .catch(error => {
            console.error('Error fetching ARIMA data:', error);
            showError('arima-chart', 'Error fetching ARIMA data');
        });
}

function renderARIMAChart(data) {
    const chartDiv = document.getElementById('arima-chart');
    if (!chartDiv) return;
    
    // Clear container and set proper styling
    chartDiv.innerHTML = '';
    chartDiv.style.display = 'block';
    chartDiv.style.width = '100%';
    chartDiv.style.height = '400px';
    chartDiv.style.minHeight = '400px';
    
    if (!data.forecast || !data.forecast.dates || data.forecast.dates.length === 0) {
        showError('arima-chart', 'No forecast data available');
        return;
    }
    
    // Prepare historical data
    const historicalDates = data.fitted_values.dates || [];
    const historicalValues = data.fitted_values.values || [];
    
    // Prepare forecast data
    const forecastDates = data.forecast.dates || [];
    const forecastValues = data.forecast.values || [];
    const forecastLower = data.forecast.lower_ci || [];
    const forecastUpper = data.forecast.upper_ci || [];
    
    const traces = [];
    
    // Historical fitted values
    if (historicalDates.length > 0 && historicalValues.length > 0) {
        traces.push({
            x: historicalDates,
            y: historicalValues,
            type: 'scatter',
            mode: 'lines',
            name: 'Fitted Values',
            line: { color: '#636efa', width: 2 }
        });
    }
    
    // Forecast
    if (forecastDates.length > 0 && forecastValues.length > 0) {
        traces.push({
            x: forecastDates,
            y: forecastValues,
            type: 'scatter',
            mode: 'lines',
            name: 'Forecast',
            line: { color: '#f59e0b', width: 2, dash: 'dash' }
        });
        
        // Confidence interval
        if (forecastLower.length > 0 && forecastUpper.length > 0) {
            traces.push({
                x: [...forecastDates, ...forecastDates.slice().reverse()],
                y: [...forecastUpper, ...forecastLower.slice().reverse()],
                type: 'scatter',
                mode: 'lines',
                name: 'Confidence Interval',
                fill: 'toself',
                fillcolor: 'rgba(245, 158, 11, 0.2)',
                line: { color: 'transparent' },
                showlegend: true
            });
        }
    }
    
    const layout = {
        title: 'ARIMA Forecast',
        xaxis: { title: 'Date' },
        yaxis: { title: 'Value' },
        hovermode: 'closest',
        plot_bgcolor: 'var(--card-bg)',
        paper_bgcolor: 'var(--card-bg)',
        font: { color: 'var(--text-primary)' },
        showlegend: true,
        legend: { x: 0, y: 1 }
    };
    
    Plotly.newPlot(chartDiv, traces, layout, {
        responsive: true, 
        displayModeBar: false,
        autosize: true
    });
    
    // Force resize after rendering
    setTimeout(() => {
        Plotly.Plots.resize(chartDiv);
    }, 100);
}

function updateARIMAStats(data) {
    if (data.aic !== null && data.aic !== undefined) {
        const aicEl = document.querySelector('#arima-chart').parentElement.querySelector('.stats-grid .stat-card:nth-child(1) .stat-value');
        if (aicEl) aicEl.textContent = data.aic.toFixed(2);
    }
    if (data.bic !== null && data.bic !== undefined) {
        const bicEl = document.querySelector('#arima-chart').parentElement.querySelector('.stats-grid .stat-card:nth-child(2) .stat-value');
        if (bicEl) bicEl.textContent = data.bic.toFixed(2);
    }
    if (data.rmse !== null && data.rmse !== undefined) {
        const rmseEl = document.querySelector('#arima-chart').parentElement.querySelector('.stats-grid .stat-card:nth-child(3) .stat-value');
        if (rmseEl) rmseEl.textContent = data.rmse.toFixed(4);
    }
}

function fetchGARCH(ticker, period) {
    fetch(`/api/time_series/garch?ticker=${ticker}&period=${period}`)
        .then(response => response.json())
        .then(data => {
            console.log('GARCH data:', data);
            if (data.status === 'success' || data.status === 'fallback') {
                renderGARCHChart(data);
                updateGARCHStats(data);
            } else {
                console.error('GARCH data error:', data.message);
                showError('garch-chart', 'Error loading GARCH data');
            }
        })
        .catch(error => {
            console.error('Error fetching GARCH data:', error);
            showError('garch-chart', 'Error fetching GARCH data');
        });
}

function renderGARCHChart(data) {
    const chartDiv = document.getElementById('garch-chart');
    if (!chartDiv) return;
    
    // Clear container and set proper styling
    chartDiv.innerHTML = '';
    chartDiv.style.display = 'block';
    chartDiv.style.width = '100%';
    chartDiv.style.height = '400px';
    chartDiv.style.minHeight = '400px';
    
    if (!data.historical_volatility || !data.historical_volatility.dates || data.historical_volatility.dates.length === 0) {
        showError('garch-chart', 'No volatility data available');
        return;
    }
    
    const dates = data.historical_volatility.dates || [];
    const volatility = data.historical_volatility.values || [];
    const rollingVol = data.historical_volatility.rolling_volatility || [];
    
    const traces = [];
    
    // Historical volatility
    if (dates.length > 0 && volatility.length > 0) {
        traces.push({
            x: dates,
            y: volatility,
            type: 'scatter',
            mode: 'lines',
            name: data.model_type === 'GARCH(1,1)' ? 'GARCH Volatility' : 'Estimated Volatility',
            line: { color: '#ef4444', width: 2 }
        });
    }
    
    // Rolling volatility (for comparison)
    if (dates.length > 0 && rollingVol.length > 0) {
        traces.push({
            x: dates,
            y: rollingVol,
            type: 'scatter',
            mode: 'lines',
            name: 'Rolling Volatility',
            line: { color: '#636efa', width: 1, dash: 'dot' }
        });
    }
    
    // Forecast volatility
    if (data.volatility_forecast && data.volatility_forecast.dates && data.volatility_forecast.dates.length > 0) {
        traces.push({
            x: data.volatility_forecast.dates,
            y: data.volatility_forecast.values,
            type: 'scatter',
            mode: 'lines',
            name: 'Volatility Forecast',
            line: { color: '#f59e0b', width: 2, dash: 'dash' }
        });
    }
    
    const layout = {
        title: 'GARCH Volatility',
        xaxis: { title: 'Date' },
        yaxis: { title: 'Annualized Volatility' },
        hovermode: 'closest',
        plot_bgcolor: 'var(--card-bg)',
        paper_bgcolor: 'var(--card-bg)',
        font: { color: 'var(--text-primary)' },
        showlegend: true,
        legend: { x: 0, y: 1 }
    };
    
    Plotly.newPlot(chartDiv, traces, layout, {
        responsive: true, 
        displayModeBar: false,
        autosize: true
    });
    
    // Force resize after rendering
    setTimeout(() => {
        Plotly.Plots.resize(chartDiv);
    }, 100);
}

function updateGARCHStats(data) {
    if (data.parameters) {
        const alphaEl = document.querySelector('#garch-chart').parentElement.querySelector('.stats-grid .stat-card:nth-child(1) .stat-value');
        if (alphaEl && data.parameters.alpha !== null) alphaEl.textContent = data.parameters.alpha.toFixed(4);
        
        const betaEl = document.querySelector('#garch-chart').parentElement.querySelector('.stats-grid .stat-card:nth-child(2) .stat-value');
        if (betaEl && data.parameters.beta !== null) betaEl.textContent = data.parameters.beta.toFixed(4);
        
        const omegaEl = document.querySelector('#garch-chart').parentElement.querySelector('.stats-grid .stat-card:nth-child(3) .stat-value');
        if (omegaEl && data.parameters.omega !== null) omegaEl.textContent = data.parameters.omega.toFixed(6);
    }
}

function updateRollingStats() {
    const ticker = document.getElementById('ts-ticker-select').value;
    const period = document.getElementById('ts-period-select').value;
    const window = document.getElementById('rolling-window').value;
    const metric = document.getElementById('rolling-metric').value;
    
    fetch(`/api/time_series/rolling_stats?ticker=${ticker}&period=${period}&window=${window}&metric=${metric}`)
        .then(response => response.json())
        .then(data => {
            console.log('Rolling stats data:', data);
            if (data.status === 'success' || data.status === 'fallback') {
                renderRollingStatsChart(data, metric);
            } else {
                console.error('Rolling stats data error:', data.message);
                showError('rolling-stats-chart', 'Error loading rolling statistics');
            }
        })
        .catch(error => {
            console.error('Error fetching rolling stats:', error);
            showError('rolling-stats-chart', 'Error fetching rolling statistics');
        });
}

function renderRollingStatsChart(data, metric) {
    const chartDiv = document.getElementById('rolling-stats-chart');
    if (!chartDiv) return;
    
    // Clear container and set proper styling
    chartDiv.innerHTML = '';
    chartDiv.style.display = 'block';
    chartDiv.style.width = '100%';
    chartDiv.style.height = '400px';
    chartDiv.style.minHeight = '400px';
    
    const traces = [];
    
    // Log returns
    if (data.log_returns && data.log_returns.dates && data.log_returns.dates.length > 0) {
        traces.push({
            x: data.log_returns.dates,
            y: data.log_returns.values,
            type: 'scatter',
            mode: 'lines',
            name: 'Log Returns',
            line: { color: '#636efa', width: 1 },
            opacity: 0.6
        });
    }
    
    // Rolling mean
    if (data.rolling_mean && data.rolling_mean.dates && data.rolling_mean.dates.length > 0) {
        traces.push({
            x: data.rolling_mean.dates,
            y: data.rolling_mean.values,
            type: 'scatter',
            mode: 'lines',
            name: 'Rolling Mean',
            line: { color: '#22c55e', width: 2 }
        });
    }
    
    // Rolling volatility
    if (data.rolling_volatility && data.rolling_volatility.dates && data.rolling_volatility.dates.length > 0) {
        traces.push({
            x: data.rolling_volatility.dates,
            y: data.rolling_volatility.values,
            type: 'scatter',
            mode: 'lines',
            name: 'Rolling Volatility (Annualized)',
            line: { color: '#ef4444', width: 2 },
            yaxis: 'y2'
        });
    }
    
    // Rolling skewness (if available)
    if (data.rolling_skewness && data.rolling_skewness.dates && data.rolling_skewness.dates.length > 0) {
        traces.push({
            x: data.rolling_skewness.dates,
            y: data.rolling_skewness.values,
            type: 'scatter',
            mode: 'lines',
            name: 'Rolling Skewness',
            line: { color: '#f59e0b', width: 2 }
        });
    }
    
    const layout = {
        title: `Rolling Statistics - ${metric.charAt(0).toUpperCase() + metric.slice(1)}`,
        xaxis: { title: 'Date' },
        yaxis: { 
            title: 'Returns / Mean / Skewness',
            side: 'left'
        },
        yaxis2: {
            title: 'Volatility (Annualized)',
            side: 'right',
            overlaying: 'y'
        },
        hovermode: 'closest',
        plot_bgcolor: 'var(--card-bg)',
        paper_bgcolor: 'var(--card-bg)',
        font: { color: 'var(--text-primary)' },
        showlegend: true,
        legend: { x: 0, y: 1 }
    };
    
    Plotly.newPlot(chartDiv, traces, layout, {
        responsive: true, 
        displayModeBar: false,
        autosize: true
    });
    
    // Force resize after rendering
    setTimeout(() => {
        Plotly.Plots.resize(chartDiv);
    }, 100);
}

function showError(chartId, message) {
    const chartDiv = document.getElementById(chartId);
    if (chartDiv) {
        chartDiv.innerHTML = '';
        chartDiv.style.display = 'flex';
        chartDiv.style.alignItems = 'center';
        chartDiv.style.justifyContent = 'center';
        chartDiv.innerHTML = `<div style="color: var(--text-secondary); text-align: center; padding: 20px;">${message}</div>`;
    }
}

