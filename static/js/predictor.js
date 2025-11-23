// Load available stocks on page load
document.addEventListener('DOMContentLoaded', function() {
    loadAvailableStocks();
    
    // Add event listeners
    document.getElementById('predict-btn').addEventListener('click', makePrediction);
});

async function loadAvailableStocks() {
    try {
        const response = await fetch('/api/stocks');
        const data = await response.json();
        
        const stockSelect = document.getElementById('stock-select');
        data.stocks.forEach(stock => {
            const option = document.createElement('option');
            option.value = stock.symbol;
            option.textContent = `${stock.symbol.replace('.NS', '')} - ${stock.name}`;
            stockSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading stocks:', error);
        showError('Failed to load available stocks');
    }
}

async function makePrediction() {
    const symbol = document.getElementById('stock-select').value;
    const days = document.getElementById('prediction-days').value;
    
    if (!symbol) {
        showError('Please select a stock');
        return;
    }
    
    // Show loading state
    showLoading();
    hideResults();
    hideError();
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                symbol: symbol,
                days: parseInt(days)
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayResults(data);
        } else {
            showError(data.error || 'Prediction failed');
        }
    } catch (error) {
        console.error('Error making prediction:', error);
        showError('Failed to connect to prediction service');
    } finally {
        hideLoading();
    }
}

function showLoading() {
    document.getElementById('loading').style.display = 'block';
}

function hideLoading() {
    document.getElementById('loading').style.display = 'none';
}

function showResults() {
    document.getElementById('results').style.display = 'block';
}

function hideResults() {
    document.getElementById('results').style.display = 'none';
}

function showError(message) {
    document.getElementById('error-text').textContent = message;
    document.getElementById('error').style.display = 'block';
}

function hideError() {
    document.getElementById('error').style.display = 'none';
}

function displayResults(data) {
    // Update stock info
    document.getElementById('current-price').textContent = `₹${data.current_price}`;
    
    const performanceElement = document.getElementById('30day-performance');
    const performance = data.stock_info.recent_performance['30_day_change_pct'];
    performanceElement.textContent = `${performance > 0 ? '+' : ''}${performance}%`;
    performanceElement.className = performance > 0 ? 'performance positive' : 'performance negative';
    
    const sentimentElement = document.getElementById('market-sentiment');
    sentimentElement.textContent = data.sentiment.sentiment_label;
    sentimentElement.className = `sentiment ${data.sentiment.sentiment_label.toLowerCase()}`;
    
    // Update confidence
    const avgConfidence = data.predictions.reduce((sum, p) => sum + p.confidence, 0) / data.predictions.length;
    document.getElementById('prediction-confidence').textContent = `${Math.round(avgConfidence * 100)}%`;
    
    // Render prediction chart
    renderPredictionChart(data);
    
    // Update predictions table
    updatePredictionsTable(data.predictions);
    
    // Update model metrics
    document.getElementById('mse').textContent = data.model_metrics.mse;
    document.getElementById('mae').textContent = data.model_metrics.mae;
    document.getElementById('rmse').textContent = data.model_metrics.rmse;
    
    showResults();
}

function renderPredictionChart(data) {
    const dates = data.prediction_dates;
    const predictions = data.predictions.map(p => p.predicted_price);
    const currentPrice = data.current_price;
    
    const upperBound = data.predictions.map(p => p.upper_bound);
    const lowerBound = data.predictions.map(p => p.lower_bound);
    
    const trace1 = {
        x: dates,
        y: predictions,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Predicted Price',
        line: {
            color: '#4f46e5',
            width: 3
        },
        marker: {
            size: 8
        }
    };
    
    const trace2 = {
        x: dates,
        y: upperBound,
        type: 'scatter',
        mode: 'lines',
        name: 'Upper Bound',
        line: {
            color: 'rgba(79, 70, 229, 0.3)',
            width: 1
        }
    };
    
    const trace3 = {
        x: dates,
        y: lowerBound,
        type: 'scatter',
        mode: 'lines',
        name: 'Lower Bound',
        fill: 'tonexty',
        fillcolor: 'rgba(79, 70, 229, 0.1)',
        line: {
            color: 'rgba(79, 70, 229, 0.3)',
            width: 1
        }
    };
    
    const currentPriceLine = {
        x: [dates[0], dates[dates.length - 1]],
        y: [currentPrice, currentPrice],
        type: 'scatter',
        mode: 'lines',
        name: 'Current Price',
        line: {
            color: '#10b981',
            width: 2,
            dash: 'dash'
        }
    };
    
    const layout = {
        title: `${data.stock_name} Price Prediction`,
        xaxis: {
            title: 'Date',
            showgrid: true,
            gridcolor: '#374151'
        },
        yaxis: {
            title: 'Price (₹)',
            showgrid: true,
            gridcolor: '#374151'
        },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {
            color: '#ffffff'
        },
        legend: {
            x: 0,
            y: 1
        }
    };
    
    const config = {
        responsive: true,
        displayModeBar: false
    };
    
    Plotly.newPlot('prediction-chart', [trace2, trace3, trace1, currentPriceLine], layout, config);
}

function updatePredictionsTable(predictions) {
    const tbody = document.querySelector('#predictions-table tbody');
    tbody.innerHTML = '';
    
    predictions.forEach(pred => {
        const row = document.createElement('tr');
        
        row.innerHTML = `
            <td>${pred.date}</td>
            <td>₹${pred.predicted_price}</td>
            <td class="${pred.price_change_pct >= 0 ? 'positive' : 'negative'}">
                ${pred.price_change_pct >= 0 ? '+' : ''}${pred.price_change_pct}%
            </td>
            <td>${Math.round(pred.confidence * 100)}%</td>
            <td>₹${pred.lower_bound} - ₹${pred.upper_bound}</td>
        `;
        
        tbody.appendChild(row);
    });
}