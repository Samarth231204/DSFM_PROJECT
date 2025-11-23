// Main JS: switch SPY/VIX flow to NIFTY/India VIX

document.addEventListener('DOMContentLoaded', function() {
    // Fetch all data when page loads
    fetchMarketSummary();
    fetchGainersLosers();
    fetchHeatmap();
    fetchSectorPerformance();
    fetchSectorRotations();
    fetchInsiderTrades();
    fetchCongressTrades();
    // Main chart: default Indian stock
    fetchStockHistory('RELIANCE.NS', '1mo');
    
    // Refresh data every 5 minutes
    setInterval(function() {
        fetchMarketSummary();
        fetchGainersLosers();
        fetchHeatmap();
        fetchSectorPerformance();
        fetchSectorRotations();
    fetchInsiderTrades();
    fetchCongressTrades();
    }, 300000); // 5 minutes
    
    // Handle window resize for all charts
    let resizeTimer;
    window.addEventListener('resize', function() {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(function() {
            const charts = ['main-chart', 'nifty-chart', 'heatmap', 'sectors-chart', 'sectors-rotation'];
            charts.forEach(chartId => {
                const chartDiv = document.getElementById(chartId);
                if (chartDiv && (chartDiv.data || chartDiv._fullLayout)) {
                    Plotly.Plots.resize(chartDiv);
                }
            });
        }, 250);
    });
});

// Update market summary call (expects nifty_data + india_vix)
function fetchMarketSummary() {
    fetch('/api/market_summary')
        .then(response => response.json())
        .then(data => {
            updateMarketOverview(data);
            // renderNiftyChart(data.nifty_data);
        })
        .catch(error => {
            console.error('Error fetching market summary:', error);
        });
}

function updateMarketOverview(data) {
    // Sentiment
    const sentimentElement = document.getElementById('sentiment-value');
    sentimentElement.textContent = data.sentiment;
    sentimentElement.className = data.sentiment.toLowerCase();

    // Market status
    const statusElement = document.getElementById('market-status');
    statusElement.textContent = data.market_status;
    statusElement.className = data.market_status === 'Open' ? 'bullish' : '';

    // NIFTY change
    const niftyElement = document.getElementById('nifty-change');
    niftyElement.textContent = data.nifty_change;
    const niftyChange = parseFloat(data.nifty_change);
    niftyElement.className = niftyChange >= 0 ? 'positive-change' : 'negative-change';

    // India VIX value
    const vixElement = document.getElementById('india-vix-value');
    vixElement.textContent = data.india_vix;
}

// Render NIFTY chart (replaces SPY chart)
function renderNiftyChart(data) {
    const trace = {
        x: data.dates,
        y: data.prices,
        type: 'scatter',
        mode: 'lines',
        line: { color: '#27ae60', width: 2 },
        fill: 'tozeroy',
        fillcolor: 'rgba(39, 174, 96, 0.1)'
    };
    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { t: 10, r: 10, b: 40, l: 40 },
        xaxis: { showgrid: false, color: '#b3b3b3' },
        yaxis: { showgrid: true, gridcolor: '#333333', color: '#b3b3b3' },
        autosize: true
    };
    const config = { responsive: true, displayModeBar: false, autosize: true };
    Plotly.newPlot('nifty-chart', [trace], layout, config);
    setTimeout(() => {
        const chartDiv = document.getElementById('nifty-chart');
        if (chartDiv) Plotly.Plots.resize(chartDiv);
    }, 100);
}

// Fetch historical data for a specific Indian stock (uses yfinance on server)
function fetchStockHistory(ticker = 'RELIANCE.NS', period = '1mo') {
    fetch(`/api/history?ticker=${encodeURIComponent(ticker)}&period=${period}`)
        .then(r => r.json())
        .then(data => {
            renderMainChart(ticker, data);
        })
        .catch(err => {
            console.error('Error fetching stock history:', err);
        });
}

function renderMainChart(ticker, data) {
    // data: { dates: [...], prices: [...] }
    const dates = data.dates || [];
    const prices = data.prices || [];

    // compute SMA(20)
    const sma = prices.map((_, i, arr) => {
        const w = arr.slice(Math.max(0, i - 19), i + 1);
        if (!w.length) return null;
        return w.reduce((a,b)=>a+b,0)/w.length;
    });

    // update ticker & price
    const latestPrice = prices.length ? prices[prices.length - 1].toFixed(2) : '--';
    document.getElementById('main-ticker').textContent = ticker.split('.')[0];
    document.getElementById('main-price').textContent = `₹${latestPrice}`;

    const traceLine = { x: dates, y: prices, mode: 'lines', line: { color: '#ffffff', width: 2 }, name: 'Price' };
    const traceSMA = { x: dates, y: sma, mode: 'lines', line: { color: '#9aa4b2', width: 1.5, dash: 'dash' }, name: 'SMA(20)' };

    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { t: 10, r: 40, b: 40, l: 40 },
        xaxis: { showgrid: false, color: '#b3b3b3' },
        yaxis: { showgrid: true, gridcolor: '#333333', color: '#b3b3b3', side: 'right' },
        showlegend: false,
        height: 420
    };

    Plotly.react('main-chart', [traceLine, traceSMA], layout, {
        responsive: true, 
        displayModeBar: false,
        autosize: true
    });
    
    // Force resize after rendering
    setTimeout(() => {
        const chartDiv = document.getElementById('main-chart');
        if (chartDiv) {
            Plotly.Plots.resize(chartDiv);
        }
    }, 100);
}

// Fetch gainers and losers data
function fetchGainersLosers() {
    fetch('/api/gainers_losers')
        .then(response => response.json())
        .then(data => {
            updateGainersTable(data.gainers);
            updateLosersTable(data.losers);
        })
        .catch(error => {
            console.error('Error fetching gainers/losers:', error);
        });
}

// Update gainers table
// Paginated table renderer used by gainers and losers
function renderPaginatedTable(items, tableSelector, infoId, pagerId, perPage = 15, role = 'gainers') {
    const tableBody = document.querySelector(`${tableSelector} tbody`);
    const infoEl = document.getElementById(infoId);
    const pagerEl = document.getElementById(pagerId);
    if (!tableBody || !infoEl || !pagerEl) return;

    let page = 1;
    const total = items.length;
    const pages = Math.max(1, Math.ceil(total / perPage));

    function drawPage(p) {
        page = Math.max(1, Math.min(p, pages));
        tableBody.innerHTML = '';
        const start = (page - 1) * perPage;
        const end = Math.min(total, start + perPage);
        for (let i = start; i < end; i++) {
            const stock = items[i];
            const row = document.createElement('tr');

            const sym = document.createElement('td');
            sym.textContent = stock.symbol.replace('.NS','');
            sym.className = 'cell-symbol';

            const price = document.createElement('td');
            price.textContent = stock.price || '--';

            const change = document.createElement('td');
            change.textContent = stock.change || '--';
            // color
            if ((stock.change || '').toString().startsWith('-')) change.className = 'negative-change';
            else change.className = 'positive-change';

            const vol = document.createElement('td');
            vol.textContent = stock.volume || '--';

            const rvol = document.createElement('td');
            rvol.textContent = stock.rvol || '--';

            const fl = document.createElement('td');
            fl.textContent = stock.float || '--';

            const mcap = document.createElement('td');
            mcap.textContent = stock.mcap || '--';

            row.appendChild(sym);
            row.appendChild(price);
            row.appendChild(change);
            row.appendChild(vol);
            row.appendChild(rvol);
            row.appendChild(fl);
            row.appendChild(mcap);

            tableBody.appendChild(row);
        }

        infoEl.textContent = `Showing ${start + 1} to ${end} of ${total} entries`;
        renderPager();
    }

    function renderPager() {
        pagerEl.innerHTML = '';
        const prev = document.createElement('button');
        prev.textContent = 'Previous';
        prev.disabled = page === 1;
        prev.className = 'pager-btn';
        prev.onclick = () => drawPage(page - 1);
        pagerEl.appendChild(prev);

        // show up to 5 page buttons centered around current
        const maxButtons = 5;
        let startPage = Math.max(1, page - Math.floor(maxButtons/2));
        let endPage = Math.min(pages, startPage + maxButtons - 1);
        if (endPage - startPage < maxButtons -1) {
            startPage = Math.max(1, endPage - maxButtons + 1);
        }

        for (let p = startPage; p <= endPage; p++) {
            const b = document.createElement('button');
            b.textContent = p;
            b.className = p === page ? 'pager-btn active' : 'pager-btn';
            b.onclick = (() => { const pp = p; return () => drawPage(pp); })();
            pagerEl.appendChild(b);
        }

        const next = document.createElement('button');
        next.textContent = 'Next';
        next.disabled = page === pages;
        next.className = 'pager-btn';
        next.onclick = () => drawPage(page + 1);
        pagerEl.appendChild(next);
    }

    // initial draw
    drawPage(1);

    // expose for debugging
    return { drawPage };
}

// Update gainers table (uses paginated renderer)
function updateGainersTable(gainers) {
    renderPaginatedTable(gainers, '#gainers-table', 'gainers-info', 'gainers-pager', 15, 'gainers');
    // keep compact market summary based on top gainers
    renderMarketSummaryList(gainers.slice(0, 7));
}

// Update losers table
function updateLosersTable(losers) {
    renderPaginatedTable(losers, '#losers-table', 'losers-info', 'losers-pager', 15, 'losers');
}

function renderMarketSummaryList(stocks) {
    const container = document.getElementById('market-summary-list');
    container.innerHTML = '';
    if (!stocks || !stocks.length) {
        container.innerHTML = '<div style="grid-column:1/-1; color:var(--text-secondary);">No data</div>';
        return;
    }
    // track selected ticker for highlight
    let selectedTicker = null;

    function highlightSelected(ticker) {
        selectedTicker = ticker;
        document.querySelectorAll('[data-ms-ticker]').forEach(el => {
            if (el.getAttribute('data-ms-ticker') === ticker) el.classList.add('active-summary');
            else el.classList.remove('active-summary');
        });
    }

    stocks.forEach(s => {
        const fullTicker = (s.symbol || s.ticker || '').toString();
        let ticker = fullTicker;
        if (!ticker.includes('.') && ticker) ticker = `${ticker}.NS`; // assume NSE if missing

        // Symbol cell
        const symDiv = document.createElement('div');
        symDiv.style.display = 'flex';
        symDiv.style.alignItems = 'center';
        symDiv.style.gap = '8px';
        symDiv.setAttribute('data-ms-ticker', ticker);
        symDiv.className = 'market-summary-item';
        const abbrev = (fullTicker.split('.')[0] || '').slice(0,2).toUpperCase();
        symDiv.innerHTML = `<div style='width:28px;height:28px;border-radius:50%;background:#1b1d20;display:flex;align-items:center;justify-content:center;color:#9aa4b2;'>${abbrev}</div><div style='font-weight:600'>${(fullTicker.split('.')[0]||fullTicker)}</div>`;

        // Price cell
        const priceDiv = document.createElement('div');
        priceDiv.style.color = 'var(--text-secondary)';
        priceDiv.setAttribute('data-ms-ticker', ticker);
        priceDiv.className = 'market-summary-item';
        priceDiv.textContent = s.price || '--';

        // Change cell with badge
        const changeDiv = document.createElement('div');
        changeDiv.style.textAlign = 'right';
        changeDiv.setAttribute('data-ms-ticker', ticker);
        changeDiv.className = 'market-summary-item';
        const changeText = s.change || '--';
        const neg = (changeText || '').toString().trim().startsWith('-');
        changeDiv.innerHTML = `<span style='padding:4px 8px;border-radius:8px;background:${neg ? 'rgba(239,68,68,0.12)' : 'rgba(16,185,129,0.12)'};color:${neg ? '#ef4444' : '#10b981'};font-weight:600'>${changeText}</span>`;

        // Click handlers: clicking any of the three elements will load the stock
        const onClick = () => {
            highlightSelected(ticker);
            fetchStockHistory(ticker, '1mo');
        };
        symDiv.style.cursor = priceDiv.style.cursor = changeDiv.style.cursor = 'pointer';
        symDiv.addEventListener('click', onClick);
        priceDiv.addEventListener('click', onClick);
        changeDiv.addEventListener('click', onClick);

        container.appendChild(symDiv);
        container.appendChild(priceDiv);
        container.appendChild(changeDiv);
    });
}

// Fetch heatmap data
function fetchHeatmap() {
    fetch('/api/heatmap')
        .then(response => response.json())
        .then(data => {
            // defensive: API may return {stocks: [...]} or an array directly
            console.log('heatmap payload:', data);
            let stocks = [];
            if (!data) {
                console.warn('No heatmap data received');
                return;
            }
            if (Array.isArray(data)) stocks = data;
            else if (Array.isArray(data.stocks)) stocks = data.stocks;
            else if (Array.isArray(data.stocks?.stocks)) stocks = data.stocks.stocks;
            else if (Array.isArray(data.items)) stocks = data.items;

            if (!stocks.length) {
                console.warn('Heatmap: no stock items to render', data);
                // clear container
                const c = document.getElementById('heatmap');
                if (c) {
                    // Fallback: create a small mock Indian stocks list so the treemap shows immediately
                    const mockSymbols = ['RELIANCE.NS','TCS.NS','INFY.NS','HDFCBANK.NS','ICICIBANK.NS','SBIN.NS','HINDUNILVR.NS','ITC.NS','LT.NS','BAJFINANCE.NS'];
                    const mockStocks = mockSymbols.map((sym, i) => ({
                        symbol: sym,
                        name: sym.split('.')[0],
                        sector: ['Energy','Technology','Financial Services','Consumer'][i%4],
                        price: Math.round((1500 + Math.random()*1500)*100)/100,
                        change: Math.round((Math.random()*4-2)*100)/100
                    }));
                    renderHeatmap(mockStocks);
                }
                return;
            }

            renderHeatmap(stocks);
        })
        .catch(error => {
            console.error('Error fetching heatmap:', error);
            const c = document.getElementById('heatmap'); if (c) c.innerHTML = '<div style="color:var(--text-secondary);padding:12px;">Heatmap load error</div>';
        });
}

// Render market heatmap
function renderHeatmap(stocks) {
    const container = document.getElementById('heatmap');
    container.innerHTML = '';

    // Unique sectors
    const sectors = [...new Set(stocks.map(s => s.sector || 'Other'))];

    const labels = [];
    const parents = [];
    const values = [];
    const text = [];
    const colors = [];

    // 1) Add root node
    labels.push('Market');
    parents.push('');
    values.push(0); 
    text.push('Market');
    colors.push('rgba(30,30,30,0.25)');

    // 2) Add sector nodes
    sectors.forEach(sec => {
        const children = stocks.filter(s => s.sector === sec);
    
        // Sum of child node values
        const sectorValue = children.reduce((acc, s) => {
            const price = typeof s.price === 'number'
                ? s.price
                : parseFloat(String(s.price).replace(/[^0-9.-]+/g, '')) || 1;
            return acc + Math.abs(price);
        }, 0);
    
        labels.push(sec);
        parents.push('Market');
        values.push(sectorValue);   // ✅ Correct parent value
        text.push(sec);
        colors.push('rgba(0,0,0,0)');
    });

    // 3) Add stock nodes
    stocks.forEach(s => {
        const sec = s.sector || 'Other';

        const price = typeof s.price === 'number'
            ? s.price
            : parseFloat(String(s.price).replace(/[^0-9.-]+/g, '')) || 1;

        const change = typeof s.change === 'number'
            ? s.change
            : parseFloat(String(s.change).replace(/[%\s]/g, '')) || 0;

            const clean = s.symbol.replace('.NS', '');

            labels.push(clean);
            parents.push(sec);
            values.push(Math.abs(price));
            text.push(`<span style="font-size:12px;">${change >= 0 ? '+' : ''}${change.toFixed(2)}%</span>`);

            colors.push(getColor(change));
    });

    // Fix root node to sum of sector values
values[0] = values
.slice(1, 1 + sectors.length) // only sector parents
.reduce((a, b) => a + b, 0);

const trace = {
    type: 'treemap',
    labels: labels,
    parents: parents,
    values: values,
    text: text,
    textinfo: "label+value+text",
hovertemplate: '<b>%{label}</b><br>%{text}<extra></extra>',
    marker: { colors: colors, line: { width: 1, color: '#111' } },
    textfont: {
        size: 14,          // stock label size
        color: '#ffffff',  // stock text color
        parent: {
            size: 16,      // sector label size
            color: '#cccccc'
        }
    },
    branchvalues: 'total'
};


    const layout = {
        margin: { t: 10, l: 10, r: 10, b: 10 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        height: 420
    };

    Plotly.newPlot(container, [trace], layout, { responsive: true, displayModeBar: false, autosize: true });
    setTimeout(() => {
        if (container) Plotly.Plots.resize(container);
    }, 100);

    // Color helper
    function getColor(change) {
        const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
        const v = clamp(change, -10, 10) / 10;
        if (v >= 0) {
            const g = Math.floor(200 - 100 * (1 - v));
            const r = Math.floor(60 - 20 * v);
            const b = Math.floor(80 - 30 * v);
            return `rgb(${r},${g},${b})`;
        } else {
            const rv = Math.abs(v);
            const r = Math.floor(200 - 50 * (1 - rv));
            const g = Math.floor(70 - 50 * rv);
            const b = Math.floor(80 - 40 * rv);
            return `rgb(${r},${g},${b})`;
        }
    }
}


// Fetch sector performance data
function fetchSectorPerformance() {
    fetch('/api/sector_performance')
        .then(response => response.json())
        .then(data => {
            renderSectorPerformance(data.sectors || []);
        })
        .catch(error => {
            console.error('Error fetching sector performance:', error);
        });
}

// Fetch sector rotations (time-series per sector)
function fetchSectorRotations() {
    fetch('/api/sector_rotations')
        .then(r => r.json())
        .then(payload => {
            if (!payload || !payload.dates || !payload.series) {
                console.warn('No sector rotations payload', payload);
                const el = document.getElementById('sectors-rotation');
                if (el) el.innerHTML = '<div style="color:var(--text-secondary);padding:12px;">No rotation data</div>';
                return;
            }
            renderSectorRotations(payload.dates, payload.series);
        })
        .catch(err => {
            console.error('Error fetching sector rotations:', err);
            const el = document.getElementById('sectors-rotation'); if (el) el.innerHTML = '<div style="color:var(--text-secondary);padding:12px;">Rotation load error</div>';
        });
}

function renderSectorRotations(dates, series) {
    // series: [{name, values:[], color?}, ...]
    const traces = series.map(s => ({
        x: dates,
        y: s.values,
        name: s.name,
        mode: 'lines+markers',
        line: { width: 2, color: s.color || undefined },
        marker: { size: 4 }
    }));

    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { t: 20, r: 20, b: 60, l: 60 },
        xaxis: { tickangle: -45, tickfont: { color: '#b3b3b3' }, showgrid: false },
        yaxis: { title: 'Stock Change (%)', color: '#b3b3b3', gridcolor: '#333' },
        legend: { orientation: 'h', y: 1.12, x: 0 },
        height: 420
    };

    Plotly.newPlot('sectors-rotation', traces, layout, {responsive: true, displayModeBar: false, autosize: true});
    setTimeout(() => {
        const chartDiv = document.getElementById('sectors-rotation');
        if (chartDiv) Plotly.Plots.resize(chartDiv);
    }, 100);
}

// Render sector performance chart
function renderSectorPerformance(sectors) {
    // defensive: if no sectors, show placeholder
    const container = document.getElementById('sectors-chart');
    if (!sectors || !sectors.length) {
        if (container) container.innerHTML = '<div style="color:var(--text-secondary);padding:12px;">No sector performance data available</div>';
        return;
    }

    // Sort sectors by performance
    sectors.sort((a, b) => b.performance - a.performance);
    
    const sectorNames = sectors.map(s => s.name);
    const performances = sectors.map(s => parseFloat(s.performance) || 0);
    const colors = performances.map(p => p >= 0 ? '#2ecc71' : '#e74c3c');
    
    const trace = {
        x: performances,
        y: sectorNames,
        type: 'bar',
        orientation: 'h',
        marker: {
            color: colors
        },
        hovertemplate: '%{y}: %{x:.2f}%<extra></extra>'
    };
    
    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { t: 10, r: 10, b: 40, l: 160 },
        xaxis: {
            title: 'Performance (%)',
            showgrid: true,
            gridcolor: '#333333',
            color: '#b3b3b3'
        },
        yaxis: {
            showgrid: false,
            color: '#b3b3b3',
            automargin: true
        },
        autosize: true,
        height: 420
    };
    
    const config = {
        responsive: true,
        displayModeBar: false,
        autosize: true
    };
    
    Plotly.newPlot('sectors-chart', [trace], layout, config);
    setTimeout(() => {
        const chartDiv = document.getElementById('sectors-chart');
        if (chartDiv) Plotly.Plots.resize(chartDiv);
    }, 100);
}

// Fetch insider trades data
function fetchInsiderTrades() {
    fetch('/api/insider_trades')
        .then(response => response.json())
        .then(data => {
            // Expect data.trades as array of insider transactions
            updateInsiderTransactionsTable(data.trades || []);
        })
        .catch(error => {
            console.error('Error fetching insider trades:', error);
        });
}

// Fetch congress trades separately
function fetchCongressTrades() {
    fetch('/api/congress_trades')
        .then(r => r.json())
        .then(data => {
            updateCongressTradesTable(data.trades || []);
        })
        .catch(err => {
            console.error('Error fetching congress trades:', err);
        });
}

// Update insider trades table
// Insider transactions: paginated custom renderer (10 per page)
function updateInsiderTransactionsTable(items) {
    const tableBody = document.querySelector('#insider-transactions-table tbody');
    const infoEl = document.getElementById('insider-info');
    const pagerEl = document.getElementById('insider-pager');
    const perPage = 10;
    let page = 1;
    const total = items.length;
    const pages = Math.max(1, Math.ceil(total / perPage));

    function draw(p) {
        page = Math.max(1, Math.min(p, pages));
        tableBody.innerHTML = '';
        const start = (page - 1) * perPage;
        const end = Math.min(total, start + perPage);
        for (let i = start; i < end; i++) {
            const t = items[i];
            const row = document.createElement('tr');
            const date = document.createElement('td'); date.textContent = t.filedAt || t.date || '--';
            const ticker = document.createElement('td'); ticker.textContent = (t.ticker || t.symbol || '--').replace('.NS','');
            const action = document.createElement('td');
            const badge = document.createElement('span'); badge.className = 'badge ' + ((t.action||t.side||'Sell').toLowerCase().includes('buy') ? 'buy' : 'sell'); badge.textContent = (t.action||t.side||'Sell');
            action.appendChild(badge);
            const shares = document.createElement('td'); shares.textContent = t.shares || t.volume || '--';
            const amount = document.createElement('td'); amount.textContent = t.amount || '--';
            row.appendChild(date); row.appendChild(ticker); row.appendChild(action); row.appendChild(shares); row.appendChild(amount);
            tableBody.appendChild(row);
        }
        infoEl.textContent = `Showing ${start+1} to ${end} of ${total} entries`;
        renderPager(pagerEl, page, pages, draw);
    }
    draw(1);
}

// Congress trades paginated renderer
function updateCongressTradesTable(items) {
    const tableBody = document.querySelector('#congress-trades-table tbody');
    const infoEl = document.getElementById('congress-info');
    const pagerEl = document.getElementById('congress-pager');
    const perPage = 10;
    let page = 1;
    const total = items.length;
    const pages = Math.max(1, Math.ceil(total / perPage));

    function draw(p) {
        page = Math.max(1, Math.min(p, pages));
        tableBody.innerHTML = '';
        const start = (page - 1) * perPage;
        const end = Math.min(total, start + perPage);
        for (let i = start; i < end; i++) {
            const t = items[i];
            const row = document.createElement('tr');
            const date = document.createElement('td'); date.textContent = t.date || '--';
            const sym = document.createElement('td'); sym.textContent = (t.symbol||t.ticker||'--').replace('.NS','');
            const amount = document.createElement('td'); amount.textContent = t.amount || '--';
            const rep = document.createElement('td'); rep.textContent = t.representative || t.rep || '--';
            row.appendChild(date); row.appendChild(sym); row.appendChild(amount); row.appendChild(rep);
            tableBody.appendChild(row);
        }
        infoEl.textContent = `Showing ${start+1} to ${end} of ${total} entries`;
        renderPager(pagerEl, page, pages, draw);
    }
    draw(1);
}

// small pager utility
function renderPager(container, current, pages, onChange) {
    container.innerHTML = '';
    const prev = document.createElement('button'); prev.textContent = 'Previous'; prev.disabled = current===1; prev.className='pager-btn'; prev.onclick = ()=> onChange(current-1);
    container.appendChild(prev);
    const maxButtons = 6;
    let start = Math.max(1, current - Math.floor(maxButtons/2));
    let end = Math.min(pages, start + maxButtons -1);
    if (end - start < maxButtons -1) start = Math.max(1, end - maxButtons +1);
    for (let p=start;p<=end;p++){ const b=document.createElement('button'); b.textContent=p; b.className = p===current ? 'pager-btn active' : 'pager-btn'; b.onclick = ((pp)=>()=>onChange(pp))(p); container.appendChild(b);}    
    const next = document.createElement('button'); next.textContent = 'Next'; next.disabled = current===pages; next.className='pager-btn'; next.onclick = ()=> onChange(current+1); container.appendChild(next);
}