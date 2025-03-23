// Initialize AOS (Animate on Scroll)
document.addEventListener('DOMContentLoaded', function () {
    // Initialize AOS
    AOS.init({
        duration: 800,
        easing: 'ease',
        once: true,
        offset: 100
    });

    // Theme toggle functionality
    const themeToggle = document.querySelector('.theme-toggle');
    const themeIcon = document.getElementById('theme-icon');

    themeToggle.addEventListener('click', function () {
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

    // Mobile menu toggle
    const mobileMenuToggle = document.querySelector('.mobile-menu-toggle');
    const navMenu = document.querySelector('nav ul');

    mobileMenuToggle.addEventListener('click', function () {
        navMenu.classList.toggle('show');
        if (navMenu.classList.contains('show')) {
            navMenu.style.display = 'flex';
            navMenu.style.flexDirection = 'column';
            navMenu.style.position = 'absolute';
            navMenu.style.top = '60px';
            navMenu.style.left = '0';
            navMenu.style.right = '0';
            navMenu.style.backgroundColor = document.body.classList.contains('dark-mode') ? 'var(--dark-bg)' : 'white';
            navMenu.style.padding = '1rem';
            navMenu.style.boxShadow = 'var(--box-shadow)';
        } else {
            navMenu.style.display = 'none';
        }
    });

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);

            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 80,
                    behavior: 'smooth'
                });

                // If mobile menu is open, close it
                if (navMenu.classList.contains('show')) {
                    navMenu.classList.remove('show');
                    navMenu.style.display = 'none';
                }

                // Update active link
                document.querySelectorAll('nav ul li a').forEach(link => {
                    link.classList.remove('active');
                });
                this.classList.add('active');
            }
        });
    });

    // Active link on scroll
    window.addEventListener('scroll', function () {
        let current = '';
        const sections = document.querySelectorAll('section');
        const scrollPosition = window.scrollY + 100;

        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;

            if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
                current = '#' + section.getAttribute('id');
            }
        });

        document.querySelectorAll('nav ul li a').forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === current) {
                link.classList.add('active');
            }
        });
    });

    // Animated crowd density visualization
    const crowdAnimation = document.querySelector('.crowd-animation');
    if (crowdAnimation) {
        createCrowdVisualization(crowdAnimation);
    }

    // Demo form submission
    const demoForm = document.querySelector('.demo-form');
    if (demoForm) {
        demoForm.addEventListener('submit', function (e) {
            e.preventDefault();

            // Basic form validation
            const inputs = this.querySelectorAll('input');
            let isValid = true;

            inputs.forEach(input => {
                if (input.value.trim() === '') {
                    input.style.borderColor = 'var(--danger)';
                    isValid = false;
                } else {
                    input.style.borderColor = 'transparent';
                }
            });

            if (isValid) {
                // Mock form submission - in real app, would send data to server
                const submitBtn = this.querySelector('button[type="submit"]');
                const originalText = submitBtn.innerText;

                submitBtn.innerText = 'Sending...';
                submitBtn.disabled = true;

                // Simulate API call
                setTimeout(() => {
                    submitBtn.innerText = 'Request Sent!';
                    submitBtn.style.backgroundColor = 'var(--success)';

                    // Reset form
                    inputs.forEach(input => input.value = '');

                    // Reset button after delay
                    setTimeout(() => {
                        submitBtn.innerText = originalText;
                        submitBtn.style.backgroundColor = '';
                        submitBtn.disabled = false;
                    }, 3000);
                }, 1500);
            }
        });
    }
});

// Create animated crowd density visualization
function createCrowdVisualization(container) {
    // This would be a placeholder for a more complex visualization
    // In a real implementation, you might use D3.js or Canvas to create
    // an actual crowd density heatmap

    // For demonstration purposes, we'll create a simple animation
    // that simulates crowd movement patterns

    const dots = [];
    const dotCount = 200; // Number of dots representing people

    // Create SVG element
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', '100%');
    svg.style.overflow = 'visible';
    container.appendChild(svg);

    // Create gradient for heatmap effect
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    svg.appendChild(defs);

    const gradient = document.createElementNS('http://www.w3.org/2000/svg', 'radialGradient');
    gradient.setAttribute('id', 'dotGradient');
    defs.appendChild(gradient);

    const stop1 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
    stop1.setAttribute('offset', '0%');
    stop1.setAttribute('stop-color', 'rgba(255, 125, 0, 0.8)');
    gradient.appendChild(stop1);

    const stop2 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
    stop2.setAttribute('offset', '100%');
    stop2.setAttribute('stop-color', 'rgba(255, 125, 0, 0)');
    gradient.appendChild(stop2);

    // Create dots
    for (let i = 0; i < dotCount; i++) {
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');

        const x = Math.random() * 100;
        const y = Math.random() * 100;
        const size = Math.random() * 5 + 2;

        circle.setAttribute('cx', x + '%');
        circle.setAttribute('cy', y + '%');
        circle.setAttribute('r', size);
        circle.setAttribute('fill', 'url(#dotGradient)');
        circle.setAttribute('opacity', Math.random() * 0.5 + 0.2);

        svg.appendChild(circle);

        dots.push({
            element: circle,
            x: x,
            y: y,
            vx: Math.random() * 0.2 - 0.1,
            vy: Math.random() * 0.2 - 0.1,
            size: size
        });
    }

    // Animate dots
    function animateDots() {
        dots.forEach(dot => {
            // Update position
            dot.x += dot.vx;
            dot.y += dot.vy;

            // Bounce off edges
            if (dot.x < 0 || dot.x > 100) dot.vx *= -1;
            if (dot.y < 0 || dot.y > 100) dot.vy *= -1;

            // Update DOM
            dot.element.setAttribute('cx', dot.x + '%');
            dot.element.setAttribute('cy', dot.y + '%');
        });

        requestAnimationFrame(animateDots);
    }

    animateDots();
}

// Add UI interactions for features that will be implemented later
function setupFutureInteractions() {
    // Create mockup heat areas for crowd visualization
    const heatMapAreas = [
        { x: 20, y: 30, radius: 15, intensity: 0.8 },
        { x: 60, y: 40, radius: 20, intensity: 0.6 },
        { x: 40, y: 70, radius: 25, intensity: 0.9 },
        { x: 80, y: 60, radius: 10, intensity: 0.5 }
    ];

    // This function would render these heat areas in a real implementation
    // Here it's just a placeholder for future development
}

// Simulate loading animation for dashboard preview
function loadDashboardPreview() {
    const dashboardPreview = document.querySelector('.dashboard-preview');
    if (dashboardPreview) {
        dashboardPreview.innerHTML = `
            <div class="loading-overlay">
                <div class="loading-spinner"></div>
                <div class="loading-text">Loading dashboard preview...</div>
            </div>
        `;

        // In a real implementation, this would load actual dashboard data
        setTimeout(() => {
            dashboardPreview.querySelector('.loading-overlay').style.opacity = 0;
        }, 2000);
    }
}

// Call this function when element is in viewport
function isElementInViewport(el) {
    const rect = el.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
}

// Add intersection observer for lazy loading and animations
if ('IntersectionObserver' in window) {
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('in-view');
                observer.unobserve(entry.target);

                // If it's the dashboard preview, load it
                if (entry.target.classList.contains('dashboard-preview')) {
                    loadDashboardPreview();
                }
            }
        });
    }, observerOptions);

    // Observe elements that should animate on scroll
    document.querySelectorAll('.feature-card, .impact-card, .dashboard-preview').forEach(el => {
        observer.observe(el);
    });
}