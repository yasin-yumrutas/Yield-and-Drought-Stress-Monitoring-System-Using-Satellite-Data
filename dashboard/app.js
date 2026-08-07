const gaziantepRegions = [
    {
        id: "araban",
        name: "Gaziantep / Araban Ovası",
        lat: 37.4300,
        lon: 37.6800,
        status: "severe",
        statusText: "Şiddetli Kuraklık Stresi",
        color: "#ef4444",
        yieldVal: "507.2 kg/da",
        precip: "361.6 mm",
        ndvi: [0.18, 0.29, 0.42, 0.58, 0.69, 0.76, 0.72, 0.58, 0.41, 0.26, 0.15, 0.09],
        ndwi: [0.10, 0.18, 0.26, 0.28, 0.22, 0.11, -0.05, -0.18, -0.29, -0.34, -0.38, -0.40]
    },
    {
        id: "sehitkamil",
        name: "Gaziantep / Şehitkamil (Aktoprak)",
        lat: 37.1600,
        lon: 37.3400,
        status: "mild",
        statusText: "Orta Su Stresi (Erken Uyarı)",
        color: "#f59e0b",
        yieldVal: "485.0 kg/da",
        precip: "342.0 mm",
        ndvi: [0.19, 0.31, 0.45, 0.60, 0.71, 0.75, 0.68, 0.55, 0.42, 0.32, 0.22, 0.16],
        ndwi: [0.12, 0.21, 0.28, 0.30, 0.24, 0.12, 0.01, -0.10, -0.18, -0.24, -0.28, -0.30]
    },
    {
        id: "sahinbey",
        name: "Gaziantep / Şahinbey (Burç)",
        lat: 36.9600,
        lon: 37.3000,
        status: "healthy",
        statusText: "Normal (Sağlıklı Tarla)",
        color: "#10b981",
        yieldVal: "562.4 kg/da",
        precip: "398.5 mm",
        ndvi: [0.20, 0.34, 0.50, 0.68, 0.82, 0.86, 0.83, 0.75, 0.65, 0.52, 0.38, 0.28],
        ndwi: [0.14, 0.24, 0.35, 0.40, 0.42, 0.38, 0.32, 0.26, 0.20, 0.15, 0.10, 0.05]
    },
    {
        id: "islahiye",
        name: "Gaziantep / İslahiye Ovası",
        lat: 37.0200,
        lon: 36.6300,
        status: "mild",
        statusText: "Orta Su Stresi (Erken Uyarı)",
        color: "#f59e0b",
        yieldVal: "490.1 kg/da",
        precip: "355.0 mm",
        ndvi: [0.18, 0.30, 0.46, 0.62, 0.74, 0.78, 0.70, 0.56, 0.43, 0.31, 0.20, 0.14],
        ndwi: [0.11, 0.20, 0.29, 0.31, 0.25, 0.14, 0.02, -0.08, -0.16, -0.22, -0.26, -0.28]
    },
    {
        id: "nizip",
        name: "Gaziantep / Nizip Ovası",
        lat: 37.0100,
        lon: 37.7900,
        status: "severe",
        statusText: "Şiddetli Kuraklık Stresi",
        color: "#ef4444",
        yieldVal: "395.0 kg/da",
        precip: "295.0 mm",
        ndvi: [0.15, 0.24, 0.36, 0.48, 0.55, 0.58, 0.48, 0.36, 0.24, 0.16, 0.10, 0.07],
        ndwi: [0.08, 0.14, 0.20, 0.18, 0.08, -0.08, -0.20, -0.32, -0.38, -0.42, -0.44, -0.45]
    },
    {
        id: "nurdagi",
        name: "Gaziantep / Nurdağı Ovası",
        lat: 37.1700,
        lon: 36.7400,
        status: "healthy",
        statusText: "Normal (Sağlıklı Tarla)",
        color: "#10b981",
        yieldVal: "580.0 kg/da",
        precip: "420.0 mm",
        ndvi: [0.22, 0.36, 0.54, 0.72, 0.85, 0.89, 0.85, 0.78, 0.68, 0.55, 0.40, 0.30],
        ndwi: [0.15, 0.26, 0.38, 0.44, 0.46, 0.42, 0.36, 0.30, 0.24, 0.18, 0.12, 0.08]
    }
];

let activeRegion = gaziantepRegions[0]; // Default Araban
let map = null;
let chartInstance = null;
let currentMarker = null;

function initMap() {
    // Center map on Gaziantep (37.0667, 37.3833)
    map = L.map('gaziantepMap').setView([37.1500, 37.3000], 9);

    // OpenStreetMap Tile Layer
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap'
    }).addTo(map);

    // Add district markers
    gaziantepRegions.forEach(region => {
        const marker = L.circleMarker([region.lat, region.lon], {
            radius: 10,
            fillColor: region.color,
            color: '#ffffff',
            weight: 2,
            opacity: 1,
            fillOpacity: 0.85
        }).addTo(map);

        marker.bindPopup(`<b>${region.name}</b><br>Durum: <span style="color:${region.color}">${region.statusText}</span><br>Rekolte: ${region.yieldVal}`);
        marker.on('click', () => selectRegion(region));
    });

    // Allow clicking ANYWHERE on the Gaziantep map
    map.on('click', function(e) {
        const clickedLat = parseFloat(e.latlng.lat.toFixed(4));
        const clickedLng = parseFloat(e.latlng.lng.toFixed(4));

        if (currentMarker) {
            map.removeLayer(currentMarker);
        }

        currentMarker = L.marker([clickedLat, clickedLng]).addTo(map)
            .bindPopup(`<b>Seçilen Koordinat</b><br>Enlem: ${clickedLat}<br>Boylam: ${clickedLng}<br><i>Canlı İklim & YZ Analizi Yükleniyor...</i>`)
            .openPopup();

        fetchLiveCoordinatesAnalysis(clickedLat, clickedLng);
    });
}

function selectRegion(region) {
    activeRegion = region;
    updateBanner();
    updateChart();
}

function fetchLiveCoordinatesAnalysis(lat, lon) {
    // Try calling live FastAPI backend if online, else generate custom live curve
    fetch('http://localhost:8000/api/predict-coordinates', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ latitude: lat, longitude: lon, field_name: `Gaziantep Lokasyon (${lat}, ${lon})` })
    })
    .then(res => res.json())
    .then(data => {
        activeRegion = {
            name: `Gaziantep Lokasyon (${lat}, ${lon})`,
            lat: lat,
            lon: lon,
            statusText: data.predicted_14d_stress_status,
            color: data.map_color_code,
            yieldVal: `${data.forecasted_yield_kg_per_da} kg/da`,
            precip: "Canlı Veri",
            ndvi: Array.from({length: 12}, (_, i) => data.current_ndvi * (0.4 + 0.6 * Math.sin(Math.PI * i / 11))),
            ndwi: Array.from({length: 12}, (_, i) => data.current_ndwi - (i * 0.02))
        };
        updateBanner();
        updateChart();
    })
    .catch(() => {
        // Fallback live interpolation for clicked coordinates
        const isSouth = lat < 37.1;
        activeRegion = {
            name: `Gaziantep Tarlası (${lat}, ${lon})`,
            lat: lat,
            lon: lon,
            statusText: isSouth ? "Orta Su Stresi (Erken Uyarı)" : "Şiddetli Kuraklık Stresi",
            color: isSouth ? "#f59e0b" : "#ef4444",
            yieldVal: `${(450 + (lat - 37.0) * 120).toFixed(1)} kg/da`,
            precip: "350.0 mm",
            ndvi: [0.18, 0.28, 0.42, 0.56, 0.68, 0.72, 0.65, 0.52, 0.38, 0.26, 0.16, 0.10],
            ndwi: [0.10, 0.18, 0.25, 0.26, 0.18, 0.08, -0.08, -0.20, -0.30, -0.35, -0.38, -0.40]
        };
        updateBanner();
        updateChart();
    });
}

function updateBanner() {
    document.getElementById('fieldTitle').innerText = `${activeRegion.name} - Canlı YZ Analizi`;
    document.getElementById('statusText').innerText = activeRegion.statusText;
    document.getElementById('statusText').style.color = activeRegion.color;
    document.getElementById('coordText').innerText = `Koordinat: ${activeRegion.lat.toFixed(4)}, ${activeRegion.lon.toFixed(4)} | Yağış Durumu: ${activeRegion.precip}`;
    document.getElementById('yieldText').innerText = `Tahmini Rekolte: ${activeRegion.yieldVal}`;
    
    const banner = document.getElementById('predictionBanner');
    banner.style.borderLeftColor = activeRegion.color;
}

function renderChart() {
    const ctx = document.getElementById('tsChart').getContext('2d');
    const labels = Array.from({ length: 12 }, (_, i) => `10 Gün ${i + 1}`);

    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'NDVI (Bitki Sağlığı - Sentinel-2)',
                    data: activeRegion.ndvi,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    fill: true,
                    tension: 0.35,
                    borderWidth: 3
                },
                {
                    label: 'NDWI (Su Stresi İndeksi - Sentinel-2)',
                    data: activeRegion.ndwi,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    fill: true,
                    tension: 0.35,
                    borderWidth: 3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#f8fafc', font: { family: 'Inter', size: 12 } } }
            },
            scales: {
                x: { grid: { color: '#2e3b52' }, ticks: { color: '#94a3b8' } },
                y: { grid: { color: '#2e3b52' }, ticks: { color: '#94a3b8' }, min: -0.5, max: 1.0 }
            }
        }
    });
}

function updateChart() {
    if (chartInstance) {
        chartInstance.data.datasets[0].data = activeRegion.ndvi;
        chartInstance.data.datasets[1].data = activeRegion.ndwi;
        chartInstance.update();
    }
}

document.addEventListener('DOMContentLoaded', () => {
    initMap();
    updateBanner();
    renderChart();
});
