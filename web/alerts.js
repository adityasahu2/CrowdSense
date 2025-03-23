document.addEventListener("DOMContentLoaded", function () {
    // Theme Toggle Functionality
    const themeToggle = document.querySelector('.theme-toggle');
    const themeIcon = document.getElementById('theme-icon');

    themeToggle.addEventListener('click', () => {
        document.body.classList.toggle('dark-mode');

        if (document.body.classList.contains('dark-mode')) {
            themeIcon.classList.remove('fa-moon');
            themeIcon.classList.add('fa-sun');
        } else {
            themeIcon.classList.remove('fa-sun');
            themeIcon.classList.add('fa-moon');
        }
    });

    // Mobile Menu Toggle
    const mobileMenuToggle = document.querySelector('.mobile-menu-toggle');
    const navMenu = document.querySelector('nav ul');

    mobileMenuToggle.addEventListener('click', () => {
        navMenu.style.display = navMenu.style.display === 'flex' ? 'none' : 'flex';
    });

    // Simulated Alert Activity
    const activityContainer = document.querySelector('.activity-alerts');
    const activities = [
        '<i class="fas fa-exclamation-circle"></i> High density detected in Zone B',
        '<i class="fas fa-temperature-high"></i> Temperature threshold exceeded',
        '<i class="fas fa-person-running"></i> Crowd movement alert triggered',
        '<i class="fas fa-bell"></i> Emergency response notified',
        '<i class="fas fa-clock"></i> Monitoring elevated for next 30 min',
        '<i class="fas fa-camera"></i> Additional cameras activated',
        '<i class="fas fa-bullhorn"></i> Alert announcement triggered',
        '<i class="fas fa-user-shield"></i> Security team dispatched'
    ];

    // Function to add new activity item
    function addNewActivity() {
        // Remove the oldest activity if we already have 4
        if (activityContainer.children.length >= 4) {
            activityContainer.removeChild(activityContainer.children[0]);
        }

        // Create new activity element
        const newActivity = document.createElement('div');
        newActivity.className = 'activity-item';
        newActivity.innerHTML = activities[Math.floor(Math.random() * activities.length)];
        newActivity.style.opacity = '0';
        newActivity.style.transform = 'translateY(10px)';

        // Add to container
        activityContainer.appendChild(newActivity);

        // Trigger animation
        setTimeout(() => {
            newActivity.style.opacity = '1';
            newActivity.style.transform = 'translateY(0)';
        }, 10);
    }

    // Simulate new activities every 5 seconds
    setInterval(addNewActivity, 5000);
});