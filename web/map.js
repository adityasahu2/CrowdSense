// map.js - Separate JavaScript file for the map functionality

// Theme Toggle Functionality
function initThemeToggle() {
    const themeToggle = document.querySelector('.theme-toggle');
    const themeIcon = document.getElementById('theme-icon');

    themeToggle.addEventListener('click', () => {
        document.body.classList.toggle('dark-mode');

        if (document.body.classList.contains('dark-mode')) {
            themeIcon.classList.remove('fa-moon');
            themeIcon.classList.add('fa-sun');
            localStorage.setItem('theme', 'dark');
        } else {
            themeIcon.classList.remove('fa-sun');
            themeIcon.classList.add('fa-moon');
            localStorage.setItem('theme', 'light');
        }
    });

    // Check for saved theme preference
    const savedTheme = localStorage.getItem('theme');

    if (savedTheme === 'dark') {
        document.body.classList.add('dark-mode');
        themeIcon.classList.remove('fa-moon');
        themeIcon.classList.add('fa-sun');
    }
}

// Mobile Menu Functionality
function initMobileMenu() {
    const mobileMenuToggle = document.querySelector('.mobile-menu-toggle');
    const navMenu = document.querySelector('nav ul');

    if (mobileMenuToggle) {
        mobileMenuToggle.addEventListener('click', () => {
            navMenu.classList.toggle('active');
        });
    }
}

// Map Functionality
function initMap() {
    // Initialize the map
    const map = L.map('crowd-map').setView([51.505, -0.09], 13);

    // Add tile layer (you can use any provider you prefer)
    const lightTileLayer = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
    const darkTileLayer = 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png';

    const tileLayer = L.tileLayer(lightTileLayer, {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);

    // Toggle map tiles based on theme
    function updateMapTiles() {
        if (document.body.classList.contains('dark-mode')) {
            tileLayer.setUrl(darkTileLayer);
        } else {
            tileLayer.setUrl(lightTileLayer);
        }
    }

    // Listen for theme changes
    const themeToggle = document.querySelector('.theme-toggle');
    themeToggle.addEventListener('click', updateMapTiles);

    // Initial check
    updateMapTiles();

    // Sample data
    const crowdData = [
        {
            id: 1,
            name: "Central Station",
            location: [51.508, -0.11],
            type: "transport",
            density: 85,
            count: 1250,
            flowRate: 120,
            lastUpdated: "Just now"
        },
        {
            id: 2,
            name: "Shopping District",
            location: [51.503, -0.09],
            type: "commercial",
            density: 55,
            count: 780,
            flowRate: 65,
            lastUpdated: "5 minutes ago"
        },
        {
            id: 3,
            name: "City Park",
            location: [51.51, -0.07],
            type: "event",
            density: 25,
            count: 350,
            flowRate: 30,
            lastUpdated: "10 minutes ago"
        },
        {
            id: 4,
            name: "Main Road Junction",
            location: [51.505, -0.09],
            type: "emergency",
            density: 45,
            count: 560,
            flowRate: 50,
            lastUpdated: "2 minutes ago"
        }
    ];

    // Layer groups for filtering
    const crowdLayers = {
        all: L.layerGroup(),
        event: L.layerGroup(),
        transport: L.layerGroup(),
        commercial: L.layerGroup(),
        emergency: L.layerGroup()
    };

    // Add all layers to map
    for (const key in crowdLayers) {
        crowdLayers[key].addTo(map);
    }

    // Create markers and circles for each data point
    const markers = {};
    const circles = {};

    crowdData.forEach(point => {
        // Determine color based on density
        let color;
        if (point.density >= 70) {
            color = '#E63946'; // High density - red
        } else if (point.density >= 40) {
            color = '#FFD166'; // Medium density - yellow
        } else {
            color = '#4DD637'; // Low density - green
        }

        // Create circle
        const circle = L.circle(point.location, {
            color: color,
            fillColor: color,
            fillOpacity: 0.5,
            radius: 300 + (point.density * 2)
        });

        // Add to appropriate layer groups
        circle.addTo(crowdLayers.all);
        circle.addTo(crowdLayers[point.type]);

        // Store reference
        circles[point.id] = circle;

        // Add click event for popup
        circle.on('click', function () {
            showPopup(point);
        });
    });

    // Popup functionality
    const locationPopup = document.getElementById('location-popup');
    const popupClose = document.getElementById('popup-close');

    function showPopup(point) {
        locationPopup.querySelector('h4').textContent = point.name;
        locationPopup.querySelector('.popup-data .data-item:nth-child(1) .data-value').textContent =
            `${point.density >= 70 ? 'High' : point.density >= 40 ? 'Medium' : 'Low'} (${point.density}%)`;
        locationPopup.querySelector('.popup-data .data-item:nth-child(2) .data-value').textContent =
            `~${point.count}`;
        locationPopup.querySelector('.popup-data .data-item:nth-child(3) .data-value').textContent =
            `${point.flowRate} people/min`;
        locationPopup.querySelector('.popup-data .data-item:nth-child(4) .data-value').textContent =
            point.lastUpdated;

        locationPopup.classList.add('active');
    }

    popupClose.addEventListener('click', function () {
        locationPopup.classList.remove('active');
    });

    // Quick location markers functionality
    const locationMarkers = document.querySelectorAll('.marker-item');

    const locations = {
        downtown: [51.508, -0.11],
        station: [51.503, -0.09],
        park: [51.51, -0.07],
        mall: [51.505, -0.09]
    };

    locationMarkers.forEach(marker => {
        marker.addEventListener('click', function () {
            const location = this.dataset.location;

            if (locations[location]) {
                map.setView(locations[location], 15);

                // Add a pulse animation to highlight the area
                const pulseCircle = L.circleMarker(locations[location], {
                    color: '#FF7D00',
                    fillColor: '#FF7D00',
                    fillOpacity: 0.3,
                    radius: 30
                }).addTo(map);

                // Animate pulse and remove
                let size = 30;
                const pulseAnimation = setInterval(() => {
                    size += 5;
                    pulseCircle.setRadius(size);
                    pulseCircle.setStyle({
                        fillOpacity: Math.max(0, 0.3 - (size - 30) / 100)
                    });

                    if (size > 100) {
                        clearInterval(pulseAnimation);
                        map.removeLayer(pulseCircle);
                    }
                }, 50);
            }
        });
    });

    // Filter functionality
    const filterButtons = document.querySelectorAll('.filter-btn');

    filterButtons.forEach(button => {
        button.addEventListener('click', function () {
            const filter = this.dataset.filter;

            // Update active button
            filterButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');

            // Hide all layers
            for (const key in crowdLayers) {
                map.removeLayer(crowdLayers[key]);
            }

            // Show selected layer
            map.addLayer(crowdLayers[filter]);
        });
    });

    // Return map instance for potential external use
    return map;
}

// Initialize everything when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', function () {
    initThemeToggle();
    initMobileMenu();
    const crowdMap = initMap();

    // Make the map responsive
    window.addEventListener('resize', function () {
        if (crowdMap) {
            crowdMap.invalidateSize();
        }
    });
});