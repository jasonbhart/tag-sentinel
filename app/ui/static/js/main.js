/**
 * Tag Sentinel UI - Main JavaScript
 * Provides core functionality for the web interface
 */

// Global application state
window.TagSentinel = {
    config: {
        apiBaseUrl: '/api',
        refreshInterval: 30000, // 30 seconds
    },
    state: {
        currentPage: null,
        isOnline: navigator.onLine,
        keyboardNavigation: false,
    }
};

// DOM ready
document.addEventListener('DOMContentLoaded', function() {
    initializeApplication();
});

/**
 * Initialize the application
 */
function initializeApplication() {
    // Set up keyboard navigation detection
    detectKeyboardNavigation();

    // Initialize mobile menu
    initializeMobileMenu();

    // Initialize tables
    initializeTables();

    // Initialize forms
    initializeForms();

    // Set up online/offline detection
    setupOnlineOfflineDetection();

    // Set up periodic data refresh
    setupPeriodicRefresh();

    console.log('Tag Sentinel UI initialized');
}

/**
 * Detect keyboard navigation for accessibility
 */
function detectKeyboardNavigation() {
    // Track if user is using keyboard
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Tab') {
            document.body.classList.add('using-keyboard');
            window.TagSentinel.state.keyboardNavigation = true;
        }
    });

    // Track if user switches to mouse
    document.addEventListener('mousedown', function() {
        document.body.classList.remove('using-keyboard');
        window.TagSentinel.state.keyboardNavigation = false;
    });
}

/**
 * Initialize mobile menu functionality
 */
function initializeMobileMenu() {
    const menuBtn = document.querySelector('.mobile-menu-btn');
    const mobileNav = document.querySelector('.mobile-nav');

    if (!menuBtn || !mobileNav) return;

    menuBtn.addEventListener('click', function() {
        const isExpanded = menuBtn.getAttribute('aria-expanded') === 'true';
        const newState = !isExpanded;

        menuBtn.setAttribute('aria-expanded', newState);
        mobileNav.setAttribute('aria-hidden', !newState);

        // Animate hamburger menu
        const lines = menuBtn.querySelectorAll('.hamburger-line');
        if (newState) {
            lines[0].style.transform = 'rotate(45deg) translate(5px, 5px)';
            lines[1].style.opacity = '0';
            lines[2].style.transform = 'rotate(-45deg) translate(7px, -6px)';
        } else {
            lines[0].style.transform = 'none';
            lines[1].style.opacity = '1';
            lines[2].style.transform = 'none';
        }
    });

    // Close mobile menu when clicking outside
    document.addEventListener('click', function(e) {
        if (!menuBtn.contains(e.target) && !mobileNav.contains(e.target)) {
            menuBtn.setAttribute('aria-expanded', 'false');
            mobileNav.setAttribute('aria-hidden', 'true');

            const lines = menuBtn.querySelectorAll('.hamburger-line');
            lines[0].style.transform = 'none';
            lines[1].style.opacity = '1';
            lines[2].style.transform = 'none';
        }
    });
}

/**
 * Initialize table functionality
 */
function initializeTables() {
    const tables = document.querySelectorAll('.table');

    tables.forEach(table => {
        // Add sortable functionality
        const headers = table.querySelectorAll('th[data-sortable]');
        headers.forEach(header => {
            header.style.cursor = 'pointer';
            header.setAttribute('tabindex', '0');
            header.setAttribute('role', 'button');
            header.setAttribute('aria-sort', 'none');

            // Add keyboard and click handlers
            header.addEventListener('click', () => sortTable(table, header));
            header.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    sortTable(table, header);
                }
            });
        });
    });
}

/**
 * Sort table by column
 */
function sortTable(table, header) {
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const columnIndex = Array.from(header.parentNode.children).indexOf(header);
    const currentSort = header.getAttribute('aria-sort');

    // Determine sort direction
    let sortDirection = 'asc';
    if (currentSort === 'ascending') {
        sortDirection = 'desc';
    }

    // Reset all headers
    table.querySelectorAll('th[aria-sort]').forEach(th => {
        th.setAttribute('aria-sort', 'none');
    });

    // Set current header sort
    header.setAttribute('aria-sort', sortDirection === 'asc' ? 'ascending' : 'descending');

    // Sort rows
    rows.sort((a, b) => {
        const aVal = a.children[columnIndex].textContent.trim();
        const bVal = b.children[columnIndex].textContent.trim();

        // Try to parse as numbers first
        const aNum = parseFloat(aVal);
        const bNum = parseFloat(bVal);

        if (!isNaN(aNum) && !isNaN(bNum)) {
            return sortDirection === 'asc' ? aNum - bNum : bNum - aNum;
        }

        // Fall back to string comparison
        return sortDirection === 'asc'
            ? aVal.localeCompare(bVal)
            : bVal.localeCompare(aVal);
    });

    // Re-append sorted rows
    rows.forEach(row => tbody.appendChild(row));

    // Announce sort to screen readers
    announceToScreenReader(`Table sorted by ${header.textContent} ${sortDirection === 'asc' ? 'ascending' : 'descending'}`);
}

/**
 * Initialize form functionality
 */
function initializeForms() {
    const forms = document.querySelectorAll('form');

    forms.forEach(form => {
        // Add validation
        form.addEventListener('submit', function(e) {
            if (!validateForm(form)) {
                e.preventDefault();
            }
        });

        // Add real-time validation
        const inputs = form.querySelectorAll('input, select, textarea');
        inputs.forEach(input => {
            input.addEventListener('blur', () => validateField(input));
            input.addEventListener('input', () => clearFieldError(input));
        });
    });
}

/**
 * Validate form
 */
function validateForm(form) {
    let isValid = true;
    const inputs = form.querySelectorAll('input[required], select[required], textarea[required]');

    inputs.forEach(input => {
        if (!validateField(input)) {
            isValid = false;
        }
    });

    return isValid;
}

/**
 * Validate individual field
 */
function validateField(field) {
    const value = field.value.trim();
    const isRequired = field.hasAttribute('required');
    let isValid = true;
    let errorMessage = '';

    // Required validation
    if (isRequired && !value) {
        isValid = false;
        errorMessage = 'This field is required';
    }

    // Email validation
    if (field.type === 'email' && value) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(value)) {
            isValid = false;
            errorMessage = 'Please enter a valid email address';
        }
    }

    // URL validation
    if (field.type === 'url' && value) {
        try {
            new URL(value);
        } catch {
            isValid = false;
            errorMessage = 'Please enter a valid URL';
        }
    }

    // Update field state
    field.setAttribute('aria-invalid', !isValid);

    if (!isValid) {
        showFieldError(field, errorMessage);
    } else {
        clearFieldError(field);
    }

    return isValid;
}

/**
 * Show field error
 */
function showFieldError(field, message) {
    clearFieldError(field);

    const errorId = field.id + '-error';
    const errorElement = document.createElement('div');
    errorElement.id = errorId;
    errorElement.className = 'form-error';
    errorElement.textContent = message;
    errorElement.setAttribute('role', 'alert');

    field.setAttribute('aria-describedby', errorId);
    field.parentNode.appendChild(errorElement);
    field.parentNode.classList.add('has-error');
}

/**
 * Clear field error
 */
function clearFieldError(field) {
    const errorId = field.id + '-error';
    const errorElement = document.getElementById(errorId);

    if (errorElement) {
        errorElement.remove();
    }

    field.removeAttribute('aria-describedby');
    field.parentNode.classList.remove('has-error');
    field.setAttribute('aria-invalid', 'false');
}

/**
 * Set up online/offline detection
 */
function setupOnlineOfflineDetection() {
    window.addEventListener('online', function() {
        window.TagSentinel.state.isOnline = true;
        announceToScreenReader('Connection restored');
        hideOfflineMessage();
    });

    window.addEventListener('offline', function() {
        window.TagSentinel.state.isOnline = false;
        announceToScreenReader('Connection lost');
        showOfflineMessage();
    });
}

/**
 * Show offline message
 */
function showOfflineMessage() {
    let offlineMsg = document.getElementById('offline-message');

    if (!offlineMsg) {
        offlineMsg = document.createElement('div');
        offlineMsg.id = 'offline-message';
        offlineMsg.className = 'alert alert-warning';
        offlineMsg.setAttribute('role', 'alert');
        offlineMsg.innerHTML = `
            <strong>You're offline</strong>
            Some features may not be available until you reconnect.
        `;

        document.body.insertBefore(offlineMsg, document.body.firstChild);
    }
}

/**
 * Hide offline message
 */
function hideOfflineMessage() {
    const offlineMsg = document.getElementById('offline-message');
    if (offlineMsg) {
        offlineMsg.remove();
    }
}

/**
 * Set up periodic data refresh
 */
function setupPeriodicRefresh() {
    // Only refresh if user is active and page is visible
    setInterval(() => {
        if (document.visibilityState === 'visible' && window.TagSentinel.state.isOnline) {
            refreshPageData();
        }
    }, window.TagSentinel.config.refreshInterval);
}

/**
 * Refresh page data
 */
function refreshPageData() {
    const refreshableElements = document.querySelectorAll('[data-refresh]');

    refreshableElements.forEach(element => {
        const endpoint = element.getAttribute('data-refresh');
        if (endpoint) {
            fetchAndUpdateElement(endpoint, element);
        }
    });
}

/**
 * Fetch data and update element
 */
async function fetchAndUpdateElement(endpoint, element) {
    try {
        element.setAttribute('aria-busy', 'true');

        // Handle absolute vs relative URLs
        const url = endpoint.startsWith('http') || endpoint.startsWith('/')
            ? endpoint  // Use absolute URLs as-is
            : window.TagSentinel.config.apiBaseUrl + '/' + endpoint;

        const response = await fetch(url);
        if (!response.ok) throw new Error('Network response was not ok');

        const data = await response.json();
        updateElementWithData(element, data);

    } catch (error) {
        console.error('Failed to refresh data:', error);
    } finally {
        element.setAttribute('aria-busy', 'false');
    }
}

/**
 * Update element with data
 */
function updateElementWithData(element, data) {
    // Handle different data refresh scenarios based on endpoint
    const endpoint = element.getAttribute('data-refresh');

    if (endpoint.endsWith('/audits')) {
        // Handle audit list data
        if (typeof window.updateRecentRunsTable === 'function') {
            window.updateRecentRunsTable(data);
        }
    } else if (endpoint === '/health' || endpoint.includes('/health')) {
        // Handle health data
        if (element.hasAttribute('data-update-text')) {
            element.textContent = data.status || 'Unknown';
        }
    } else {
        // Default behavior
        if (element.hasAttribute('data-update-text')) {
            element.textContent = data.value || data.text || '';
        }

        if (element.hasAttribute('data-update-html')) {
            element.innerHTML = data.html || '';
        }
    }

    // Trigger custom update event
    element.dispatchEvent(new CustomEvent('data-updated', { detail: data }));
}

/**
 * Announce message to screen readers
 */
function announceToScreenReader(message) {
    const announcer = document.getElementById('sr-announcements');
    if (announcer) {
        announcer.textContent = message;

        // Clear after announcement
        setTimeout(() => {
            announcer.textContent = '';
        }, 1000);
    }
}

/**
 * Utility function to format dates
 */
function formatDate(dateString, options = {}) {
    const date = new Date(dateString);
    const defaultOptions = {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    };

    return date.toLocaleDateString('en-US', { ...defaultOptions, ...options });
}

/**
 * Utility function to format duration
 */
function formatDuration(milliseconds) {
    const seconds = Math.floor(milliseconds / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (hours > 0) {
        return `${hours}h ${minutes % 60}m`;
    } else if (minutes > 0) {
        return `${minutes}m ${seconds % 60}s`;
    } else {
        return `${seconds}s`;
    }
}

/**
 * Utility function to debounce function calls
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * API helper functions
 */
window.TagSentinel.api = {
    /**
     * Make authenticated API request
     */
    async request(endpoint, options = {}) {
        const url = window.TagSentinel.config.apiBaseUrl + endpoint;
        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        };

        try {
            const response = await fetch(url, config);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    },

    /**
     * Get audit runs
     */
    async getRuns(params = {}) {
        const query = new URLSearchParams(params).toString();
        return this.request(`/audits${query ? '?' + query : ''}`);
    },

    /**
     * Get audit run details
     */
    async getRunDetails(runId) {
        return this.request(`/audits/${runId}`);
    },

    /**
     * Get audit details (alias for getRunDetails)
     */
    async getAuditDetail(auditId) {
        return this.getRunDetails(auditId);
    },

    /**
     * Get page details
     */
    async getPageDetails(runId, pageId) {
        return this.request(`/audits/${runId}/pages/${pageId}`);
    }
};

// Export for use in other scripts
window.TagSentinel.utils = {
    formatDate,
    formatDuration,
    debounce,
    announceToScreenReader,
    validateForm,
    validateField
};