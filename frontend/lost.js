document.addEventListener('DOMContentLoaded', function () {
    // Theme Toggle Functionality
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

    // Person Card Selection
    const personCards = document.querySelectorAll('.person-card');

    personCards.forEach(card => {
        card.addEventListener('click', function () {
            // Remove active class from all cards
            personCards.forEach(c => c.classList.remove('active'));

            // Add active class to clicked card
            this.classList.add('active');

            // Here you would normally fetch and display person details
            // For demo, we'll just use fixed data from the HTML
            updatePersonDetail(this);
        });
    });

    function updatePersonDetail(card) {
        const name = card.querySelector('h3').textContent;
        const lastSeen = card.querySelector('p:first-of-type').textContent.replace('Last seen: ', '');
        const ageAndCase = card.querySelector('p:nth-of-type(2)').textContent.split(' • ');
        const age = ageAndCase[0].replace('Age: ', '');
        const caseId = ageAndCase[1];

        // Update detail panel elements
        document.querySelector('.detail-info h2').textContent = name;
        document.querySelector('.detail-info p:first-of-type').textContent = `${getGender(name)}, ${age} years old`;
        document.querySelector('.detail-info p:nth-of-type(2)').textContent = `Last seen: March ${getRandomDay()}, 2025 at ${lastSeen}`;
        document.querySelector('.detail-info p:nth-of-type(3)').textContent = `Case ID: ${caseId}`;

        // Update status based on tags
        const tags = card.querySelectorAll('.person-tag');
        let statusClass = '';
        let statusText = '';

        tags.forEach(tag => {
            if (tag.classList.contains('tag-resolved')) {
                statusClass = 'status-resolved';
                statusText = 'Case Resolved';
            } else if (tag.classList.contains('tag-urgent')) {
                statusClass = 'status-urgent';
                statusText = 'Urgent Investigation';
            } else if (tag.classList.contains('tag-active')) {
                statusClass = 'status-active';
                statusText = 'Active Investigation';
            }
        });

        const statusElement = document.querySelector('.detail-status');
        statusElement.className = 'detail-status'; // Reset classes
        statusElement.classList.add(statusClass);
        statusElement.textContent = statusText;
    }

    // Helper function to guess gender based on name (simplified)
    function getGender(name) {
        const femaleNames = ['Sarah', 'Emma', 'Olivia', 'Sophia'];
        return femaleNames.some(femaleName => name.includes(femaleName)) ? 'Female' : 'Male';
    }

    // Generate random day for demo purposes
    function getRandomDay() {
        return Math.floor(Math.random() * 28) + 1;
    }

    // Filter Buttons
    const filterButtons = document.querySelectorAll('.filter-btn');

    filterButtons.forEach(button => {
        button.addEventListener('click', function () {
            // Remove active class from all buttons
            filterButtons.forEach(btn => btn.classList.remove('active'));

            // Add active class to clicked button
            this.classList.add('active');

            // Filter the person cards
            const filter = this.textContent.trim().toLowerCase();
            filterPersonCards(filter);
        });
    });

    function filterPersonCards(filter) {
        personCards.forEach(card => {
            const tags = Array.from(card.querySelectorAll('.person-tag')).map(tag => tag.textContent.toLowerCase());

            if (filter === 'all') {
                card.style.display = 'flex';
            } else if (filter === 'urgent' && tags.includes('urgent')) {
                card.style.display = 'flex';
            } else if (filter === 'recent' && tags.includes('recent')) {
                card.style.display = 'flex';
            } else if (filter === 'resolved' && tags.includes('resolved')) {
                card.style.display = 'flex';
            } else {
                card.style.display = 'none';
            }
        });
    }

    // Detail tabs
    const detailTabs = document.querySelectorAll('.detail-tab');

    detailTabs.forEach(tab => {
        tab.addEventListener('click', function () {
            // Remove active class from all tabs
            detailTabs.forEach(t => t.classList.remove('active'));

            // Add active class to clicked tab
            this.classList.add('active');

            // Here you would normally show/hide content based on selected tab
            // For demo purposes, we'll keep showing the Information tab
        });
    });

    // Search functionality
    const searchInput = document.querySelector('.search-bar input');

    searchInput.addEventListener('input', function () {
        const searchTerm = this.value.toLowerCase();

        personCards.forEach(card => {
            const name = card.querySelector('h3').textContent.toLowerCase();
            const location = card.querySelector('p:first-of-type').textContent.toLowerCase();
            const caseId = card.querySelector('p:nth-of-type(2)').textContent.toLowerCase();

            if (name.includes(searchTerm) || location.includes(searchTerm) || caseId.includes(searchTerm)) {
                card.style.display = 'flex';
            } else {
                card.style.display = 'none';
            }
        });
    });

    // New Case Button
    const newCaseBtn = document.querySelector('.new-case-btn');

    newCaseBtn.addEventListener('click', function () {
        alert('New case form would open here');
        // In a real app, you would open a modal or navigate to a new case form
    });

    // Action Buttons
    const actionButtons = document.querySelectorAll('.action-btn');

    actionButtons.forEach(button => {
        button.addEventListener('click', function () {
            const action = this.textContent.trim();
            alert(`${action} action would be performed here`);
            // In a real app, you would perform the corresponding action
        });
    });

    // Window resize handler for responsive design
    window.addEventListener('resize', function () {
        if (window.innerWidth > 768 && navMenu.classList.contains('show')) {
            navMenu.classList.remove('show');
            navMenu.style.display = '';
        }
    });
});