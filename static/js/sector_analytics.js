// Sector Analytics JavaScript - Stub implementation

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all sector analytics sections
    initializeSectorAnalytics();
    
    // Set up event listeners for controls
    setupEventListeners();
    
    // Handle window resize for charts
    let resizeTimer;
    window.addEventListener('resize', function() {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(function() {
            const charts = ['correlation-matrix-chart', 'sector-returns-chart', 'sector-volatility-chart', 
                          'pca-scatter-chart', 'pca-variance-chart', 'clustering-chart', 
                          'rmt-eigenvalue-chart', 'cross-sectional-chart'];
            charts.forEach(chartId => {
                const chartDiv = document.getElementById(chartId);
                if (chartDiv && chartDiv.data) {
                    Plotly.Plots.resize(chartDiv);
                }
            });
        }, 250);
    });
});

function initializeSectorAnalytics() {
    // Fetch all sector analytics data
    fetchSectorOverview();
    fetchSectorReturns();
    fetchSectorVolatility();
    fetchCorrelationMatrix();
    fetchPCA();
    fetchClustering();
    fetchRMT();
    fetchSectorSentiment();
    fetchSectorRotation();
    fetchSectorInsights();
    fetchCrossSectional();
}

function setupEventListeners() {
    // Returns period and metric controls
    const returnsPeriod = document.getElementById('returns-period');
    const returnsMetric = document.getElementById('returns-metric');
    if (returnsPeriod && returnsMetric) {
        returnsPeriod.addEventListener('change', fetchSectorReturns);
        returnsMetric.addEventListener('change', fetchSectorReturns);
    }
    
    // Correlation method control
    const correlationMethod = document.getElementById('correlation-method');
    if (correlationMethod) {
        correlationMethod.addEventListener('change', fetchCorrelationMatrix);
    }
    
    // Clustering controls
    const clusteringAlgorithm = document.getElementById('clustering-algorithm');
    const nClusters = document.getElementById('n-clusters');
    if (clusteringAlgorithm && nClusters) {
        clusteringAlgorithm.addEventListener('change', fetchClustering);
        nClusters.addEventListener('change', fetchClustering);
    }
}

function fetchSectorOverview() {
    const period = document.getElementById('returns-period')?.value || '1y';
    
    fetch(`/api/sector_analytics/overview?period=${period}`)
        .then(response => response.json())
        .then(data => {
            console.log('Sector overview data:', data);
            if (data.status === 'success') {
                renderSectorOverview(data);
            }
        })
        .catch(error => {
            console.error('Error fetching sector overview:', error);
        });
}

function renderSectorOverview(data) {
    // Update sector returns and volatility charts with overview data
    if (data.sectors && data.sectors.length > 0) {
        // Render sector returns chart
        const returnsChartDiv = document.getElementById('sector-returns-chart');
        if (returnsChartDiv) {
            renderSectorReturnsChart(data.sectors, returnsChartDiv);
        }
        
        // Render sector volatility chart
        const volatilityChartDiv = document.getElementById('sector-volatility-chart');
        if (volatilityChartDiv) {
            renderSectorVolatilityChart(data.sectors, volatilityChartDiv);
        }
    }
}

function renderSectorReturnsChart(sectors, chartDiv) {
    chartDiv.innerHTML = '';
    chartDiv.style.display = 'block';
    chartDiv.style.width = '100%';
    chartDiv.style.height = '400px';
    chartDiv.style.minHeight = '400px';
    
    const traces = sectors.map(sector => ({
        x: sector.daily_returns.dates,
        y: sector.daily_returns.values,
        type: 'scatter',
        mode: 'lines',
        name: sector.sector,
        line: { width: 1.5 }
    }));
    
    const layout = {
        title: 'Sector Daily Returns',
        xaxis: { title: 'Date' },
        yaxis: { title: 'Daily Return' },
        hovermode: 'x unified',
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
    
    setTimeout(() => Plotly.Plots.resize(chartDiv), 100);
}

function renderSectorVolatilityChart(sectors, chartDiv) {
    chartDiv.innerHTML = '';
    chartDiv.style.display = 'block';
    chartDiv.style.width = '100%';
    chartDiv.style.height = '400px';
    chartDiv.style.minHeight = '400px';
    
    const sectorNames = sectors.map(s => s.sector);
    const volatilities = sectors.map(s => s.volatility);
    const colors = volatilities.map(v => v > 0.3 ? '#ef4444' : v > 0.2 ? '#f59e0b' : '#22c55e');
    
    const trace = {
        x: sectorNames,
        y: volatilities,
        type: 'bar',
        marker: { color: colors },
        text: volatilities.map(v => (v * 100).toFixed(2) + '%'),
        textposition: 'outside'
    };
    
    const layout = {
        title: 'Sector Volatility (Annualized)',
        xaxis: { title: 'Sector', tickangle: -45 },
        yaxis: { title: 'Volatility' },
        plot_bgcolor: 'var(--card-bg)',
        paper_bgcolor: 'var(--card-bg)',
        font: { color: 'var(--text-primary)' }
    };
    
    Plotly.newPlot(chartDiv, [trace], layout, {
        responsive: true,
        displayModeBar: false,
        autosize: true
    });
    
    setTimeout(() => Plotly.Plots.resize(chartDiv), 100);
    
    // Update volatility statistics
    if (volatilities.length > 0) {
        const maxVol = Math.max(...volatilities);
        const minVol = Math.min(...volatilities);
        const avgVol = volatilities.reduce((a, b) => a + b, 0) / volatilities.length;
        const spread = maxVol - minVol;
        
        // Find sectors with max and min volatility
        const maxSector = sectorNames[volatilities.indexOf(maxVol)];
        const minSector = sectorNames[volatilities.indexOf(minVol)];
        
        // Update the stat cards in the volatility section
        const volatilitySection = chartDiv.closest('.card');
        if (volatilitySection) {
            const statCards = volatilitySection.querySelectorAll('.stat-card');
            if (statCards.length >= 4) {
                statCards[0].querySelector('.stat-value').textContent = `${(maxVol * 100).toFixed(2)}% (${maxSector})`;
                statCards[1].querySelector('.stat-value').textContent = `${(minVol * 100).toFixed(2)}% (${minSector})`;
                statCards[2].querySelector('.stat-value').textContent = `${(avgVol * 100).toFixed(2)}%`;
                statCards[3].querySelector('.stat-value').textContent = `${(spread * 100).toFixed(2)}%`;
            }
        }
    }
}

function fetchSectorReturns() {
    // This is now handled by fetchSectorOverview
    fetchSectorOverview();
}

function fetchSectorVolatility() {
    // This is now handled by fetchSectorOverview
    fetchSectorOverview();
}

function fetchSectorSentiment() {
    const period = document.getElementById('returns-period')?.value || '1y';
    
    fetch(`/api/sector_analytics/sentiment?period=${period}`)
        .then(response => response.json())
        .then(data => {
            console.log('Sector sentiment data:', data);
            console.log('Status:', data.status);
            console.log('Sectors found:', data.sector_sentiment ? Object.keys(data.sector_sentiment).length : 0);
            console.log('Sector names:', data.sector_sentiment ? Object.keys(data.sector_sentiment) : []);
            
            // Render if we have sentiment data (success or fallback)
            if (data.sector_sentiment && Object.keys(data.sector_sentiment).length > 0) {
                renderSectorSentiment(data);
            } else {
                console.warn('No sector sentiment data available');
                const chartDiv = document.getElementById('sector-sentiment-chart');
                if (chartDiv) {
                    chartDiv.innerHTML = '<p style="padding: 20px; text-align: center; color: var(--text-secondary);">No sentiment data available. ' + (data.note || '') + '</p>';
                }
            }
        })
        .catch(error => {
            console.error('Error fetching sector sentiment:', error);
            const chartDiv = document.getElementById('sector-sentiment-chart');
            if (chartDiv) {
                chartDiv.innerHTML = '<p style="padding: 20px; text-align: center; color: var(--text-error);">Error loading sentiment data</p>';
            }
        });
}

function renderSectorSentiment(data) {
    const chartDiv = document.getElementById('sector-sentiment-chart');
    if (!chartDiv || !data.sector_sentiment) return;
    
    chartDiv.innerHTML = '';
    chartDiv.style.display = 'block';
    chartDiv.style.width = '100%';
    chartDiv.style.height = '400px';
    chartDiv.style.minHeight = '400px';
    
    const traces = [];
    const sectors = Object.keys(data.sector_sentiment);
    
    console.log(`Rendering sentiment for ${sectors.length} sectors:`, sectors);
    
    sectors.forEach(sector => {
        const sentiment = data.sector_sentiment[sector];
        if (!sentiment || !sentiment.dates || !sentiment.values) {
            console.warn(`Missing data for sector ${sector}:`, sentiment);
            return;
        }
        
        // Ensure dates and values are arrays and have the same length
        const dates = Array.isArray(sentiment.dates) ? sentiment.dates : [];
        const values = Array.isArray(sentiment.values) ? sentiment.values : [];
        
        if (dates.length === 0 || values.length === 0) {
            console.warn(`Empty data for sector ${sector}: dates=${dates.length}, values=${values.length}`);
            return;
        }
        
        // Ensure dates and values have the same length
        const minLength = Math.min(dates.length, values.length);
        const trimmedDates = dates.slice(0, minLength);
        const trimmedValues = values.slice(0, minLength);
        
        console.log(`Sector ${sector}: ${trimmedDates.length} data points, first date: ${trimmedDates[0]}, last date: ${trimmedDates[trimmedDates.length - 1]}`);
        console.log(`Value range: ${Math.min(...trimmedValues).toFixed(3)} to ${Math.max(...trimmedValues).toFixed(3)}`);
        
        traces.push({
            x: trimmedDates,
            y: trimmedValues,
            type: 'scatter',
            mode: 'lines',
            name: sector,
            line: { width: 1.5 }
        });
    });
    
    if (traces.length === 0) {
        chartDiv.innerHTML = '<p style="padding: 20px; text-align: center; color: var(--text-secondary);">No valid sentiment data to display</p>';
        return;
    }
    
    const layout = {
        title: 'Sector Sentiment Over Time',
        xaxis: { 
            title: 'Date',
            type: 'date'
        },
        yaxis: { 
            title: 'Sentiment Score',
            range: [-1, 1]
        },
        hovermode: 'x unified',
        plot_bgcolor: 'var(--card-bg)',
        paper_bgcolor: 'var(--card-bg)',
        font: { color: 'var(--text-primary)' },
        showlegend: true,
        legend: { 
            x: 0, 
            y: 1,
            orientation: 'v',
            xanchor: 'left',
            yanchor: 'top'
        },
        shapes: [{
            type: 'line',
            x0: 0,
            x1: 1,
            xref: 'paper',
            y0: 0,
            y1: 0,
            yref: 'y',
            line: { color: 'var(--border)', width: 1, dash: 'dash' }
        }]
    };
    
    Plotly.newPlot(chartDiv, traces, layout, {
        responsive: true,
        displayModeBar: false,
        autosize: true
    });
    
    setTimeout(() => Plotly.Plots.resize(chartDiv), 100);
}

function fetchSectorRotation() {
    const period = document.getElementById('returns-period')?.value || '1y';
    
    fetch(`/api/sector_analytics/rotation?period=${period}`)
        .then(response => response.json())
        .then(data => {
            console.log('Sector rotation data:', data);
            if (data.status === 'success') {
                renderSectorRotation(data);
            }
        })
        .catch(error => {
            console.error('Error fetching sector rotation:', error);
        });
}

function renderSectorRotation(data) {
    // Update rotation display if container exists
    const rotationContainer = document.getElementById('sector-rotation-display');
    if (rotationContainer && data.insights) {
        rotationContainer.innerHTML = `
            <h3>Sector Rotation Insights</h3>
            <div style="margin-top: 1rem;">
                <div><strong>Rising Sectors:</strong> ${data.rising_sectors.join(', ') || 'None'}</div>
                <div style="margin-top: 0.5rem;"><strong>Falling Sectors:</strong> ${data.falling_sectors.join(', ') || 'None'}</div>
                <ul style="margin-top: 1rem; padding-left: 1.5rem;">
                    ${data.insights.map(insight => `<li>${insight}</li>`).join('')}
                </ul>
            </div>
        `;
    }
}

function fetchSectorInsights() {
    const period = document.getElementById('returns-period')?.value || '1y';
    
    fetch(`/api/sector_analytics/insights?period=${period}`)
        .then(response => response.json())
        .then(data => {
            console.log('Sector insights data:', data);
            if (data.status === 'success') {
                renderSectorInsights(data);
            }
        })
        .catch(error => {
            console.error('Error fetching sector insights:', error);
        });
}

function renderSectorInsights(data) {
    const insightsContainer = document.getElementById('sector-insights-display');
    if (insightsContainer && data.insights) {
        insightsContainer.innerHTML = `
            <h3>Key Insights</h3>
            <ul style="margin-top: 1rem; padding-left: 1.5rem;">
                ${data.insights.map(insight => `<li style="margin-bottom: 0.5rem;">${insight}</li>`).join('')}
            </ul>
        `;
    }
}

function fetchCorrelationMatrix() {
    const method = document.getElementById('correlation-method')?.value || 'pearson';
    
    fetch(`/api/sector_analytics/correlation?method=${method}`)
        .then(response => response.json())
        .then(data => {
            console.log('Correlation matrix data:', data);
            if (data.status === 'success' || data.status === 'fallback') {
                renderCorrelationMatrix(data);
            } else {
                console.error('Correlation matrix error:', data.message);
                showSectorError('correlation-matrix-chart', 'Error loading correlation matrix');
            }
        })
        .catch(error => {
            console.error('Error fetching correlation matrix:', error);
            showSectorError('correlation-matrix-chart', 'Error fetching correlation matrix');
        });
}

function renderCorrelationMatrix(data) {
    const chartDiv = document.getElementById('correlation-matrix-chart');
    if (!chartDiv) return;
    
    // Clear container and set proper styling
    chartDiv.innerHTML = '';
    chartDiv.style.display = 'block';
    chartDiv.style.width = '100%';
    chartDiv.style.height = '500px';
    chartDiv.style.minHeight = '500px';
    
    if (!data.correlation_matrix || !data.symbols || data.correlation_matrix.length === 0) {
        showSectorError('correlation-matrix-chart', 'No correlation data available');
        return;
    }
    
    const matrix = data.correlation_matrix;
    const symbols = data.symbols;
    
    // Create heatmap trace
    const trace = {
        z: matrix,
        x: symbols,
        y: symbols,
        type: 'heatmap',
        colorscale: [
            [0, '#ef4444'],  // Red for negative
            [0.5, '#f59e0b'], // Orange for zero
            [1, '#22c55e']   // Green for positive
        ],
        zmid: 0,
        colorbar: {
            title: 'Correlation',
            titleside: 'right'
        },
        text: matrix.map(row => row.map(val => val.toFixed(3))),
        texttemplate: '%{text}',
        textfont: { size: 10 },
        hovertext: matrix.map((row, i) => 
            row.map((val, j) => `${symbols[i]} vs ${symbols[j]}: ${val.toFixed(3)}`)
        ),
        hoverinfo: 'text'
    };
    
    const layout = {
        title: `${data.method.charAt(0).toUpperCase() + data.method.slice(1)} Correlation Matrix`,
        xaxis: { 
            title: 'Symbol',
            side: 'bottom',
            tickangle: -45
        },
        yaxis: { 
            title: 'Symbol',
            autorange: 'reversed'
        },
        width: null,
        height: 500,
        margin: { l: 150, r: 50, t: 50, b: 150 },
        plot_bgcolor: 'var(--card-bg)',
        paper_bgcolor: 'var(--card-bg)',
        font: { color: 'var(--text-primary)' }
    };
    
    Plotly.newPlot(chartDiv, [trace], layout, {
        responsive: true,
        displayModeBar: false,
        autosize: true
    });
    
    // Force resize after rendering
    setTimeout(() => {
        Plotly.Plots.resize(chartDiv);
    }, 100);
}

function showSectorError(chartId, message) {
    const chartDiv = document.getElementById(chartId);
    if (chartDiv) {
        chartDiv.innerHTML = '';
        chartDiv.style.display = 'flex';
        chartDiv.style.alignItems = 'center';
        chartDiv.style.justifyContent = 'center';
        chartDiv.innerHTML = `<div style="color: var(--text-secondary); text-align: center; padding: 20px;">${message}</div>`;
    }
}

function fetchPCA() {
    fetch('/api/sector_analytics/pca')
        .then(response => response.json())
        .then(data => {
            console.log('PCA data:', data);
            if (data.status === 'success' || data.status === 'fallback') {
                renderPCAScatter(data);
                renderPCAVariance(data);
            } else {
                console.error('PCA data error:', data.message);
                showSectorError('pca-scatter-chart', 'Error loading PCA data');
                showSectorError('pca-variance-chart', 'Error loading PCA data');
            }
        })
        .catch(error => {
            console.error('Error fetching PCA data:', error);
            showSectorError('pca-scatter-chart', 'Error fetching PCA data');
            showSectorError('pca-variance-chart', 'Error fetching PCA data');
        });
}

function renderPCAScatter(data) {
    const chartDiv = document.getElementById('pca-scatter-chart');
    if (!chartDiv) return;
    
    // Clear container and set proper styling
    chartDiv.innerHTML = '';
    chartDiv.style.display = 'block';
    chartDiv.style.width = '100%';
    chartDiv.style.height = '400px';
    chartDiv.style.minHeight = '400px';
    
    if (!data.pca_2d_projection || !data.pca_2d_projection.symbols || data.pca_2d_projection.symbols.length === 0) {
        showSectorError('pca-scatter-chart', 'No PCA projection data available');
        return;
    }
    
    const symbols = data.pca_2d_projection.symbols;
    const x = data.pca_2d_projection.x;
    const y = data.pca_2d_projection.y;
    
    // Create scatter plot for PCA
    const tracePCA = {
        x: x,
        y: y,
        mode: 'markers+text',
        type: 'scatter',
        name: 'PCA Projection',
        text: symbols.map(s => s.replace('.NS', '')),
        textposition: 'top center',
        textfont: { size: 10, color: 'var(--text-primary)' },
        marker: {
            size: 8,
            color: '#636efa',
            line: { width: 1, color: 'var(--border)' }
        }
    };
    
    const traces = [tracePCA];
    
    // Add MDS projection if available
    if (data.mds_2d_projection && data.mds_2d_projection.symbols && data.mds_2d_projection.symbols.length > 0) {
        traces.push({
            x: data.mds_2d_projection.x,
            y: data.mds_2d_projection.y,
            mode: 'markers',
            type: 'scatter',
            name: 'MDS Projection',
            marker: {
                size: 6,
                color: '#ef4444',
                symbol: 'diamond',
                line: { width: 1, color: 'var(--border)' }
            }
        });
    }
    
    const layout = {
        title: 'PCA 2D Projection' + (data.mds_2d_projection ? ' vs MDS' : ''),
        xaxis: { 
            title: 'PC1',
            zeroline: true,
            zerolinecolor: 'var(--border)'
        },
        yaxis: { 
            title: 'PC2',
            zeroline: true,
            zerolinecolor: 'var(--border)'
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
    
    setTimeout(() => {
        Plotly.Plots.resize(chartDiv);
    }, 100);
}

function renderPCAVariance(data) {
    const chartDiv = document.getElementById('pca-variance-chart');
    if (!chartDiv) return;
    
    // Clear container and set proper styling
    chartDiv.innerHTML = '';
    chartDiv.style.display = 'block';
    chartDiv.style.width = '100%';
    chartDiv.style.height = '400px';
    chartDiv.style.minHeight = '400px';
    
    if (!data.explained_variance_ratio || data.explained_variance_ratio.length === 0) {
        showSectorError('pca-variance-chart', 'No variance data available');
        return;
    }
    
    const n_components = data.explained_variance_ratio.length;
    const components = Array.from({length: n_components}, (_, i) => `PC${i + 1}`);
    
    // Individual explained variance
    const traceIndividual = {
        x: components,
        y: data.explained_variance_ratio,
        type: 'bar',
        name: 'Individual',
        marker: { color: '#636efa' }
    };
    
    // Cumulative explained variance
    const cumulative = data.cumulative_variance_ratio || [];
    const traceCumulative = {
        x: components,
        y: cumulative.length > 0 ? cumulative : data.explained_variance_ratio.map((_, i, arr) => 
            arr.slice(0, i + 1).reduce((a, b) => a + b, 0)
        ),
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Cumulative',
        yaxis: 'y2',
        line: { color: '#ef4444', width: 2 },
        marker: { color: '#ef4444', size: 6 }
    };
    
    const layout = {
        title: 'Explained Variance by Principal Component',
        xaxis: { 
            title: 'Principal Component',
            tickangle: -45
        },
        yaxis: { 
            title: 'Individual Explained Variance Ratio',
            side: 'left',
            range: [0, Math.max(...data.explained_variance_ratio) * 1.1]
        },
        yaxis2: {
            title: 'Cumulative Explained Variance Ratio',
            side: 'right',
            overlaying: 'y',
            range: [0, 1.1]
        },
        plot_bgcolor: 'var(--card-bg)',
        paper_bgcolor: 'var(--card-bg)',
        font: { color: 'var(--text-primary)' },
        showlegend: true,
        legend: { x: 0, y: 1 },
        barmode: 'group'
    };
    
    Plotly.newPlot(chartDiv, [traceIndividual, traceCumulative], layout, {
        responsive: true,
        displayModeBar: false,
        autosize: true
    });
    
    setTimeout(() => {
        Plotly.Plots.resize(chartDiv);
    }, 100);
    
    // Update variance stats
    if (data.explained_variance_ratio && data.explained_variance_ratio.length > 0) {
        const pc1El = document.querySelector('#pca-variance-chart').parentElement.querySelector('.stats-grid .stat-card:nth-child(1) .stat-value');
        if (pc1El) pc1El.textContent = (data.explained_variance_ratio[0] * 100).toFixed(2) + '%';
        
        const pc2El = document.querySelector('#pca-variance-chart').parentElement.querySelector('.stats-grid .stat-card:nth-child(2) .stat-value');
        if (pc2El && data.explained_variance_ratio.length > 1) {
            pc2El.textContent = (data.explained_variance_ratio[1] * 100).toFixed(2) + '%';
        }
    }
}

function fetchClustering() {
    const algorithm = document.getElementById('clustering-algorithm')?.value || 'kmeans';
    const nClusters = document.getElementById('n-clusters')?.value || 3;
    
    fetch(`/api/sector_analytics/clustering?algorithm=${algorithm}&n_clusters=${nClusters}`)
        .then(response => response.json())
        .then(data => {
            console.log('Clustering data:', data);
            if (data.status === 'success' || data.status === 'fallback') {
                renderClusteringChart(data);
            } else {
                console.error('Clustering data error:', data.message);
                showSectorError('clustering-chart', 'Error loading clustering data');
            }
        })
        .catch(error => {
            console.error('Error fetching clustering data:', error);
            showSectorError('clustering-chart', 'Error fetching clustering data');
        });
}

function renderClusteringChart(data) {
    const chartDiv = document.getElementById('clustering-chart');
    if (!chartDiv) return;
    
    // Clear container and set proper styling
    chartDiv.innerHTML = '';
    chartDiv.style.display = 'block';
    chartDiv.style.width = '100%';
    chartDiv.style.height = '400px';
    chartDiv.style.minHeight = '400px';
    
    if (!data.pca_coordinates || data.pca_coordinates.length === 0) {
        showSectorError('clustering-chart', 'No clustering data available');
        return;
    }
    
    const pcaCoords = data.pca_coordinates;
    const nClusters = data.n_clusters || 1;
    
    // Color palette for clusters
    const colors = [
        '#636efa', '#ef553b', '#00cc96', '#ab63fa', '#ffa15a',
        '#19d3f3', '#ff6692', '#b6e880', '#ff97ff', '#fecb52'
    ];
    
    // Group coordinates by cluster
    const clusters = {};
    pcaCoords.forEach(coord => {
        const clusterId = coord.cluster;
        if (!clusters[clusterId]) {
            clusters[clusterId] = [];
        }
        clusters[clusterId].push(coord);
    });
    
    // Create traces for each cluster
    const traces = [];
    Object.keys(clusters).sort((a, b) => parseInt(a) - parseInt(b)).forEach((clusterId, idx) => {
        const clusterData = clusters[clusterId];
        const clusterSummary = data.cluster_summaries ? 
            data.cluster_summaries.find(s => s.cluster_id === parseInt(clusterId)) : null;
        
        const clusterName = clusterSummary ? 
            `Cluster ${clusterId}: ${clusterSummary.regime_type}` : 
            `Cluster ${clusterId}`;
        
        traces.push({
            x: clusterData.map(c => c.x),
            y: clusterData.map(c => c.y),
            mode: 'markers+text',
            type: 'scatter',
            name: clusterName,
            text: clusterData.map(c => c.symbol.replace('.NS', '')),
            textposition: 'top center',
            textfont: { size: 9, color: colors[idx % colors.length] },
            marker: {
                size: 10,
                color: colors[idx % colors.length],
                line: { width: 1.5, color: 'white' },
                opacity: 0.8
            },
            hovertemplate: '<b>%{text}</b><br>' +
                          'PC1: %{x:.3f}<br>' +
                          'PC2: %{y:.3f}<br>' +
                          'Return: %{customdata[0]:.2%}<br>' +
                          'Volatility: %{customdata[1]:.2%}<extra></extra>',
            customdata: clusterData.map(c => [c.mean_return, c.volatility])
        });
    });
    
    const layout = {
        title: `Market Regime Clustering (${nClusters} Clusters)`,
        xaxis: { 
            title: 'PC1 (First Principal Component)',
            zeroline: true,
            zerolinecolor: 'var(--border)'
        },
        yaxis: { 
            title: 'PC2 (Second Principal Component)',
            zeroline: true,
            zerolinecolor: 'var(--border)'
        },
        hovermode: 'closest',
        plot_bgcolor: 'var(--card-bg)',
        paper_bgcolor: 'var(--card-bg)',
        font: { color: 'var(--text-primary)' },
        showlegend: true,
        legend: { 
            x: 0, 
            y: 1,
            font: { size: 10 }
        }
    };
    
    Plotly.newPlot(chartDiv, traces, layout, {
        responsive: true,
        displayModeBar: false,
        autosize: true
    });
    
    setTimeout(() => {
        Plotly.Plots.resize(chartDiv);
    }, 100);
    
    // Update cluster summaries display
    if (data.cluster_summaries && data.cluster_summaries.length > 0) {
        const summaryContainer = chartDiv.parentElement;
        if (summaryContainer) {
            let summaryDiv = summaryContainer.querySelector('.cluster-summaries');
            if (!summaryDiv) {
                summaryDiv = document.createElement('div');
                summaryDiv.className = 'cluster-summaries';
                summaryDiv.style.cssText = 'margin-top: 1rem; padding: 1rem; background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px;';
                summaryContainer.appendChild(summaryDiv);
            }
            
            summaryDiv.innerHTML = `
                <h3 style="margin-top: 0; margin-bottom: 1rem;">Cluster Summaries</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
                    ${data.cluster_summaries.map(summary => {
                        const color = colors[summary.cluster_id % colors.length];
                        return `
                        <div style="padding: 0.75rem; background: var(--card-bg); border-left: 3px solid ${color}; border-radius: 4px; border: 1px solid var(--border);">
                            <h4 style="margin: 0 0 0.5rem 0; font-size: 0.95rem; color: ${color};">Cluster ${summary.cluster_id}: ${summary.regime_type}</h4>
                            <p style="margin: 0 0 0.5rem 0; font-size: 0.85rem; color: var(--text-secondary);">${summary.description}</p>
                            <div style="font-size: 0.8rem; color: var(--text-secondary);">
                                <div><strong>Count:</strong> ${summary.count} stocks</div>
                                <div><strong>Avg Return:</strong> ${(summary.avg_return * 100).toFixed(2)}%</div>
                                <div><strong>Avg Volatility:</strong> ${(summary.avg_volatility * 100).toFixed(2)}%</div>
                            </div>
                        </div>
                    `;
                    }).join('')}
                </div>
            `;
        }
    }
}

function fetchRMT() {
    fetch('/api/sector_analytics/rmt')
        .then(response => response.json())
        .then(data => {
            console.log('RMT data:', data);
            if (data.status === 'success' || data.status === 'fallback') {
                renderRMTEigenvalue(data);
                updateRMTInsights(data);
                if (data.raw_correlation_matrix && data.denoised_correlation_matrix) {
                    renderRMTCorrelationMatrix(data.raw_correlation_matrix, data.symbols, 'rmt-raw-correlation-chart', 'Raw Correlation Matrix');
                    renderRMTCorrelationMatrix(data.denoised_correlation_matrix, data.symbols, 'rmt-denoised-correlation-chart', 'Denoised Correlation Matrix');
                }
            } else {
                console.error('RMT data error:', data.message);
                showSectorError('rmt-eigenvalue-chart', 'Error loading RMT data');
                showSectorError('rmt-raw-correlation-chart', 'Error loading correlation matrix');
                showSectorError('rmt-denoised-correlation-chart', 'Error loading correlation matrix');
                updateRMTInsights(data);
            }
        })
        .catch(error => {
            console.error('Error fetching RMT data:', error);
            showSectorError('rmt-eigenvalue-chart', 'Error fetching RMT data');
        });
}

function updateRMTInsights(data) {
    const insightsDiv = document.getElementById('rmt-insights');
    if (!insightsDiv) return;
    
    if (data.status === 'error' || !data.eigenvalue_distribution) {
        insightsDiv.innerHTML = `
            <h3 style="margin-top: 0;">RMT Insights</h3>
            <p style="color: var(--text-secondary);">${data.message || 'No data available'}</p>
        `;
        return;
    }
    
    const noiseThreshold = data.noise_threshold !== null && data.noise_threshold !== undefined 
        ? data.noise_threshold.toFixed(4) : '--';
    const significantCount = data.significant_eigenvalues ? data.significant_eigenvalues.length : 0;
    const marketMode = data.market_mode && data.market_mode.eigenvalue 
        ? data.market_mode.eigenvalue.toFixed(4) : '--';
    const sectorModes = data.sector_modes && data.sector_modes.length > 0
        ? data.sector_modes.map(m => m.toFixed(4)).join(', ') : '--';
    const filteredInfo = data.denoised_correlation_matrix && data.raw_correlation_matrix
        ? `${data.symbols ? data.symbols.length : 0}×${data.symbols ? data.symbols.length : 0} matrix` : '--';
    
    insightsDiv.innerHTML = `
        <h3 style="margin-top: 0;">RMT Insights</h3>
        <ul style="list-style: none; padding: 0;">
            <li style="margin-bottom: 0.5rem;"><strong>Noise Threshold (MP Upper):</strong> ${noiseThreshold}</li>
            <li style="margin-bottom: 0.5rem;"><strong>Significant Eigenvalues:</strong> ${significantCount}</li>
            <li style="margin-bottom: 0.5rem;"><strong>Market Mode (λ₁):</strong> ${marketMode}</li>
            <li style="margin-bottom: 0.5rem;"><strong>Sector Modes:</strong> ${sectorModes}</li>
            <li style="margin-bottom: 0.5rem;"><strong>Denoised Matrix:</strong> ${filteredInfo}</li>
        </ul>
        ${data.statistics ? `
            <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--border);">
                <h4 style="margin-top: 0; font-size: 0.9rem;">Variance Explained</h4>
                <ul style="list-style: none; padding: 0; font-size: 0.85rem;">
                    <li style="margin-bottom: 0.3rem;">Market Mode: ${(data.statistics.variance_explained_by_market * 100).toFixed(2)}%</li>
                    ${data.statistics.variance_explained_by_sectors > 0 ? 
                        `<li style="margin-bottom: 0.3rem;">Sector Modes: ${(data.statistics.variance_explained_by_sectors * 100).toFixed(2)}%</li>` : ''}
                </ul>
            </div>
        ` : ''}
    `;
}

function renderRMTEigenvalue(data) {
    const chartDiv = document.getElementById('rmt-eigenvalue-chart');
    if (!chartDiv) return;
    
    // Clear container and set proper styling
    chartDiv.innerHTML = '';
    chartDiv.style.display = 'block';
    chartDiv.style.width = '100%';
    chartDiv.style.height = '350px';
    chartDiv.style.minHeight = '350px';
    
    if (!data.eigenvalue_distribution || !data.eigenvalue_distribution.eigenvalues || 
        data.eigenvalue_distribution.eigenvalues.length === 0) {
        showSectorError('rmt-eigenvalue-chart', 'No eigenvalue data available');
        return;
    }
    
    const eigenvalues = data.eigenvalue_distribution.eigenvalues;
    const lambda_min = data.eigenvalue_distribution.marchenko_pastur_min;
    const lambda_max = data.eigenvalue_distribution.marchenko_pastur_max;
    const significant_indices = data.eigenvalue_distribution.significant_indices || [];
    const noise_indices = data.eigenvalue_distribution.noise_indices || [];
    
    // Create index array for x-axis
    const indices = Array.from({length: eigenvalues.length}, (_, i) => i + 1);
    
    // Separate significant and noise eigenvalues
    const significant_ev = eigenvalues.map((ev, idx) => significant_indices.includes(idx) ? ev : null);
    const noise_ev = eigenvalues.map((ev, idx) => noise_indices.includes(idx) ? ev : null);
    
    // Eigenvalue spectrum
    const traceEigenvalues = {
        x: indices,
        y: eigenvalues,
        mode: 'markers+lines',
        type: 'scatter',
        name: 'Eigenvalues',
        marker: {
            size: 6,
            color: eigenvalues.map((ev, idx) => significant_indices.includes(idx) ? '#22c55e' : '#ef4444'),
            line: { width: 1, color: 'var(--border)' }
        },
        line: { color: '#636efa', width: 1 }
    };
    
    const traces = [traceEigenvalues];
    
    // Add Marchenko-Pastur bounds if available
    if (lambda_max !== null && lambda_max !== undefined) {
        // Upper bound line
        traces.push({
            x: [Math.min(...indices), Math.max(...indices)],
            y: [lambda_max, lambda_max],
            mode: 'lines',
            type: 'scatter',
            name: 'MP Upper Bound',
            line: { color: '#f59e0b', width: 2, dash: 'dash' }
        });
    }
    
    if (lambda_min !== null && lambda_min !== undefined && lambda_min > 0) {
        // Lower bound line
        traces.push({
            x: [Math.min(...indices), Math.max(...indices)],
            y: [lambda_min, lambda_min],
            mode: 'lines',
            type: 'scatter',
            name: 'MP Lower Bound',
            line: { color: '#f59e0b', width: 2, dash: 'dash' }
        });
    }
    
    // Add statistics text
    const statsText = data.statistics ? 
        `Significant: ${data.statistics.significant_count} | Noise: ${data.statistics.noise_count}` : '';
    
    const layout = {
        title: 'Eigenvalue Distribution (RMT Analysis)' + (statsText ? ` - ${statsText}` : ''),
        xaxis: { 
            title: 'Eigenvalue Index',
            type: 'linear'
        },
        yaxis: { 
            title: 'Eigenvalue',
            type: 'linear'
        },
        hovermode: 'closest',
        plot_bgcolor: 'var(--card-bg)',
        paper_bgcolor: 'var(--card-bg)',
        font: { color: 'var(--text-primary)' },
        showlegend: true,
        legend: { x: 0, y: 1 },
        annotations: significant_indices.length > 0 ? [{
            x: significant_indices[0] + 1,
            y: eigenvalues[significant_indices[0]],
            text: 'Market Mode',
            showarrow: true,
            arrowhead: 2,
            ax: 0,
            ay: -40,
            bgcolor: 'rgba(34, 197, 94, 0.2)',
            bordercolor: '#22c55e'
        }] : []
    };
    
    Plotly.newPlot(chartDiv, traces, layout, {
        responsive: true,
        displayModeBar: false,
        autosize: true
    });
    
    setTimeout(() => {
        Plotly.Plots.resize(chartDiv);
    }, 100);
    
    // Update RMT statistics if available
    if (data.statistics) {
        const statsContainer = document.querySelector('#rmt-eigenvalue-chart').parentElement;
        if (statsContainer) {
            // Update or create stats display
            let statsDiv = statsContainer.querySelector('.rmt-stats');
            if (!statsDiv) {
                statsDiv = document.createElement('div');
                statsDiv.className = 'rmt-stats';
                statsDiv.style.cssText = 'margin-top: 10px; padding: 10px; background: var(--card-bg); border-radius: 6px; font-size: 12px;';
                statsContainer.appendChild(statsDiv);
            }
            
            statsDiv.innerHTML = `
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;">
                    <div><strong>Total Eigenvalues:</strong> ${data.statistics.total_eigenvalues}</div>
                    <div><strong>Significant:</strong> ${data.statistics.significant_count} (above MP bound)</div>
                    <div><strong>Noise:</strong> ${data.statistics.noise_count} (within MP bound)</div>
                    <div><strong>Market Mode:</strong> ${(data.statistics.variance_explained_by_market * 100).toFixed(2)}%</div>
                    ${data.statistics.variance_explained_by_sectors > 0 ? 
                        `<div><strong>Sector Modes:</strong> ${(data.statistics.variance_explained_by_sectors * 100).toFixed(2)}%</div>` : ''}
                </div>
            `;
        }
    }
}

function fetchCrossSectional() {
    const period = document.getElementById('returns-period')?.value || '1y';
    
    fetch(`/api/sector_analytics/cross_sectional?period=${period}`)
        .then(response => response.json())
        .then(data => {
            console.log('Cross-sectional data:', data);
            if (data.status === 'success') {
                renderCrossSectionalChart(data);
                renderCrossSectionalTable(data);
            } else {
                console.error('Cross-sectional data error:', data.message);
                showSectorError('cross-sectional-chart', 'Error loading cross-sectional data');
            }
        })
        .catch(error => {
            console.error('Error fetching cross-sectional data:', error);
            showSectorError('cross-sectional-chart', 'Error fetching cross-sectional data');
        });
}

function renderCrossSectionalChart(data) {
    const chartDiv = document.getElementById('cross-sectional-chart');
    if (!chartDiv || !data.scatter_data) return;
    
    chartDiv.innerHTML = '';
    chartDiv.style.display = 'block';
    chartDiv.style.width = '100%';
    chartDiv.style.height = '400px';
    chartDiv.style.minHeight = '400px';
    
    const scatterData = data.scatter_data;
    if (!scatterData.sectors || scatterData.sectors.length === 0) {
        showSectorError('cross-sectional-chart', 'No cross-sectional data available');
        return;
    }
    
    // Create scatter plot: Return vs Volatility
    const trace = {
        x: scatterData.volatilities,
        y: scatterData.returns,
        mode: 'markers+text',
        type: 'scatter',
        text: scatterData.sectors,
        textposition: 'top center',
        textfont: { size: 10 },
        marker: {
            size: 12,
            color: scatterData.sharpe_ratios,
            colorscale: 'Viridis',
            showscale: true,
            colorbar: { title: 'Sharpe Ratio' },
            line: { width: 1, color: 'white' }
        },
        hovertemplate: '<b>%{text}</b><br>' +
                      'Volatility: %{x:.2%}<br>' +
                      'Return: %{y:.2%}<br>' +
                      'Sharpe: %{marker.color:.2f}<extra></extra>'
    };
    
    const layout = {
        title: 'Cross-Sectional Analysis: Return vs Volatility',
        xaxis: { 
            title: 'Volatility (Annualized)',
            tickformat: '.0%'
        },
        yaxis: { 
            title: 'Total Return',
            tickformat: '.0%'
        },
        hovermode: 'closest',
        plot_bgcolor: 'var(--card-bg)',
        paper_bgcolor: 'var(--card-bg)',
        font: { color: 'var(--text-primary)' }
    };
    
    Plotly.newPlot(chartDiv, [trace], layout, {
        responsive: true,
        displayModeBar: false,
        autosize: true
    });
    
    setTimeout(() => Plotly.Plots.resize(chartDiv), 100);
}

function renderCrossSectionalTable(data) {
    const tableBody = document.querySelector('#cross-sectional-chart').parentElement.querySelector('tbody');
    if (!tableBody || !data.sectors || data.sectors.length === 0) return;
    
    tableBody.innerHTML = data.sectors.map(sector => `
        <tr>
            <td>${sector.sector}</td>
            <td>${(sector.return * 100).toFixed(2)}%</td>
            <td>${(sector.volatility * 100).toFixed(2)}%</td>
            <td>${sector.sharpe_ratio.toFixed(2)}</td>
            <td>${sector.beta.toFixed(2)}</td>
            <td>${(sector.alpha * 100).toFixed(2)}%</td>
        </tr>
    `).join('');
}

function renderRMTCorrelationMatrix(matrix, symbols, chartId, title) {
    const chartDiv = document.getElementById(chartId);
    if (!chartDiv) return;
    
    // Clear container and set proper styling
    chartDiv.innerHTML = '';
    chartDiv.style.display = 'block';
    chartDiv.style.width = '100%';
    chartDiv.style.height = '400px';
    chartDiv.style.minHeight = '400px';
    
    if (!matrix || !symbols || matrix.length === 0 || symbols.length === 0) {
        showSectorError(chartId, 'No correlation matrix data available');
        return;
    }
    
    // Ensure matrix dimensions match symbols
    const n = Math.min(matrix.length, symbols.length);
    const matrixData = matrix.slice(0, n).map(row => row.slice(0, n));
    const symbolLabels = symbols.slice(0, n).map(s => s.replace('.NS', ''));
    
    // Create heatmap trace
    const trace = {
        z: matrixData,
        x: symbolLabels,
        y: symbolLabels,
        type: 'heatmap',
        colorscale: [
            [0, '#ef4444'],  // Red for negative
            [0.5, '#f59e0b'], // Orange for zero
            [1, '#22c55e']   // Green for positive
        ],
        zmid: 0,
        colorbar: {
            title: 'Correlation',
            titleside: 'right'
        },
        text: matrixData.map(row => row.map(val => val.toFixed(3))),
        texttemplate: '%{text}',
        textfont: { size: 8 },
        hovertext: matrixData.map((row, i) => 
            row.map((val, j) => `${symbolLabels[i]} vs ${symbolLabels[j]}: ${val.toFixed(3)}`)
        ),
        hoverinfo: 'text'
    };
    
    const layout = {
        title: title,
        xaxis: { 
            title: 'Symbol',
            side: 'bottom',
            tickangle: -45
        },
        yaxis: { 
            title: 'Symbol',
            autorange: 'reversed'
        },
        width: null,
        height: 400,
        margin: { l: 120, r: 50, t: 50, b: 120 },
        plot_bgcolor: 'var(--card-bg)',
        paper_bgcolor: 'var(--card-bg)',
        font: { color: 'var(--text-primary)' }
    };
    
    Plotly.newPlot(chartDiv, [trace], layout, {
        responsive: true,
        displayModeBar: false,
        autosize: true
    });
    
    setTimeout(() => {
        Plotly.Plots.resize(chartDiv);
    }, 100);
}

