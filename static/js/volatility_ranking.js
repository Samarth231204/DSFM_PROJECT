// Volatility Ranking JavaScript

document.addEventListener('DOMContentLoaded', function() {
    fetchVolatilityRanking();
    
    const periodSelect = document.getElementById('volatility-period');
    if (periodSelect) {
        periodSelect.addEventListener('change', fetchVolatilityRanking);
    }
    
    // Handle window resize
    let resizeTimer;
    window.addEventListener('resize', function() {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(function() {
            // Resize any charts if needed
        }, 250);
    });
});

function fetchVolatilityRanking() {
    const period = document.getElementById('volatility-period')?.value || '3mo';
    
    fetch(`/api/volatility_ranking?period=${period}`)
        .then(response => response.json())
        .then(data => {
            console.log('Volatility ranking data:', data);
            if (data.status === 'success') {
                renderVolatilityRanking(data);
            } else {
                showError('Data unavailable. Try again later.');
            }
        })
        .catch(error => {
            console.error('Error fetching volatility ranking:', error);
            showError('Error loading data. Please try again.');
        });
}

function renderVolatilityRanking(data) {
    renderMostVolatile(data.most_volatile || []);
    renderLeastVolatile(data.least_volatile || []);
}

function renderMostVolatile(stocks) {
    const container = document.getElementById('most-volatile-table');
    if (!container) return;
    
    if (stocks.length === 0) {
        container.innerHTML = '<div style="padding: 20px; text-align: center; color: var(--text-secondary);">No data available</div>';
        return;
    }
    
    const table = document.createElement('table');
    table.className = 'market-table';
    table.style.width = '100%';
    
    // Header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    ['Rank', 'Symbol', 'Sector', 'Volatility (%)'].forEach(text => {
        const th = document.createElement('th');
        th.textContent = text;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    // Body
    const tbody = document.createElement('tbody');
    stocks.forEach((stock, index) => {
        const row = document.createElement('tr');
        
        const rankCell = document.createElement('td');
        rankCell.textContent = index + 1;
        rankCell.style.fontWeight = '700';
        rankCell.style.color = '#ef4444';
        
        const symbolCell = document.createElement('td');
        symbolCell.textContent = stock.symbol;
        symbolCell.style.fontWeight = '600';
        
        const sectorCell = document.createElement('td');
        sectorCell.textContent = stock.sector;
        sectorCell.style.color = 'var(--text-secondary)';
        
        const volCell = document.createElement('td');
        volCell.textContent = stock.volatility_percent.toFixed(2) + '%';
        volCell.style.fontWeight = '700';
        volCell.style.color = '#ef4444';
        
        row.appendChild(rankCell);
        row.appendChild(symbolCell);
        row.appendChild(sectorCell);
        row.appendChild(volCell);
        tbody.appendChild(row);
    });
    table.appendChild(tbody);
    
    container.innerHTML = '';
    container.appendChild(table);
}

function renderLeastVolatile(stocks) {
    const container = document.getElementById('least-volatile-table');
    if (!container) return;
    
    if (stocks.length === 0) {
        container.innerHTML = '<div style="padding: 20px; text-align: center; color: var(--text-secondary);">No data available</div>';
        return;
    }
    
    const table = document.createElement('table');
    table.className = 'market-table';
    table.style.width = '100%';
    
    // Header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    ['Rank', 'Symbol', 'Sector', 'Volatility (%)'].forEach(text => {
        const th = document.createElement('th');
        th.textContent = text;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    // Body
    const tbody = document.createElement('tbody');
    stocks.forEach((stock, index) => {
        const row = document.createElement('tr');
        
        const rankCell = document.createElement('td');
        rankCell.textContent = index + 1;
        rankCell.style.fontWeight = '700';
        rankCell.style.color = '#22c55e';
        
        const symbolCell = document.createElement('td');
        symbolCell.textContent = stock.symbol;
        symbolCell.style.fontWeight = '600';
        
        const sectorCell = document.createElement('td');
        sectorCell.textContent = stock.sector;
        sectorCell.style.color = 'var(--text-secondary)';
        
        const volCell = document.createElement('td');
        volCell.textContent = stock.volatility_percent.toFixed(2) + '%';
        volCell.style.fontWeight = '700';
        volCell.style.color = '#22c55e';
        
        row.appendChild(rankCell);
        row.appendChild(symbolCell);
        row.appendChild(sectorCell);
        row.appendChild(volCell);
        tbody.appendChild(row);
    });
    table.appendChild(tbody);
    
    container.innerHTML = '';
    container.appendChild(table);
}

function showError(message) {
    const containers = ['most-volatile-table', 'least-volatile-table'];
    containers.forEach(id => {
        const container = document.getElementById(id);
        if (container) {
            container.innerHTML = `<div style="padding: 20px; text-align: center; color: var(--text-secondary);">${message}</div>`;
        }
    });
}

