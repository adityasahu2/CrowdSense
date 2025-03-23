// Contact Form Functionality
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
    // Initialize AOS library
    AOS.init({
        duration: 800,
        easing: 'ease-in-out',
        once: true
    });

    // Form handling
    const contactForm = document.getElementById('contactForm');
    const successMessage = document.querySelector('.success-message');
    const errorMessage = document.querySelector('.error-message');

    if (contactForm) {
        contactForm.addEventListener('submit', function (e) {
            e.preventDefault();

            // Add loading state to button
            const submitBtn = contactForm.querySelector('.submit-btn');
            const originalBtnText = submitBtn.innerHTML;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sending...';
            submitBtn.disabled = true;

            // Simulate form submission (replace with actual API call)
            setTimeout(function () {
                // For demo purposes: randomly succeed or fail
                const success = Math.random() > 0.2; // 80% success rate

                if (success) {
                    // Show success message
                    successMessage.style.display = 'block';
                    errorMessage.style.display = 'none';

                    // Reset form
                    contactForm.reset();

                    // Hide success message after 5 seconds
                    setTimeout(function () {
                        successMessage.style.display = 'none';
                    }, 5000);
                } else {
                    // Show error message
                    errorMessage.style.display = 'block';
                    successMessage.style.display = 'none';

                    // Hide error message after 5 seconds
                    setTimeout(function () {
                        errorMessage.style.display = 'none';
                    }, 5000);
                }

                // Reset button
                submitBtn.innerHTML = originalBtnText;
                submitBtn.disabled = false;
            }, 2000);
        });
    }

    // Form animation enhancements
    const formControls = document.querySelectorAll('.form-control');

    formControls.forEach(control => {
        // Add focus animations
        control.addEventListener('focus', function () {
            this.parentElement.classList.add('focused');
        });

        control.addEventListener('blur', function () {
            if (this.value === '') {
                this.parentElement.classList.remove('focused');
            }
        });

        // Check if field has value on load
        if (control.value !== '') {
            control.parentElement.classList.add('focused');
        }
    });

    // Add hover effects to info cards
    const infoCards = document.querySelectorAll('.info-card');

    infoCards.forEach(card => {
        card.addEventListener('mouseenter', function () {
            this.style.transform = 'translateY(-5px)';
            this.style.boxShadow = '0 8px 25px rgba(0, 0, 0, 0.15)';

            const icon = this.querySelector('.icon-container');
            if (icon) {
                icon.style.transform = 'scale(1.1)';
            }
        });

        card.addEventListener('mouseleave', function () {
            this.style.transform = 'translateY(0)';
            this.style.boxShadow = 'var(--box-shadow)';

            const icon = this.querySelector('.icon-container');
            if (icon) {
                icon.style.transform = 'scale(1)';
            }
        });
    });
});