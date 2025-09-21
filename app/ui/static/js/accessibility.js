/**
 * Tag Sentinel UI - Accessibility JavaScript
 * Enhanced accessibility features and keyboard navigation
 */

// Accessibility configuration
const A11Y_CONFIG = {
    // Keyboard navigation
    focusableSelectors: [
        'a[href]',
        'button:not([disabled])',
        'input:not([disabled])',
        'select:not([disabled])',
        'textarea:not([disabled])',
        '[tabindex]:not([tabindex="-1"])',
        '[contenteditable="true"]'
    ].join(', '),

    // Screen reader announcements
    announceDelay: 100,

    // Focus management
    focusOutlineClass: 'focus-visible',

    // Live regions
    liveRegionId: 'sr-announcements'
};

// Initialize accessibility features when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    initializeAccessibilityFeatures();
});

/**
 * Initialize all accessibility features
 */
function initializeAccessibilityFeatures() {
    setupKeyboardNavigation();
    setupFocusManagement();
    setupLiveRegions();
    setupModalAccessibility();
    setupTableAccessibility();
    setupFormAccessibility();
    setupTooltipAccessibility();

    console.log('Accessibility features initialized');
}

/**
 * Enhanced keyboard navigation
 */
function setupKeyboardNavigation() {
    // Trap focus in modals
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            closeTopModal();
        }

        if (e.key === 'Tab') {
            handleTabNavigation(e);
        }
    });

    // Add keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Skip if user is typing in form field
        if (e.target.matches('input, textarea, select, [contenteditable]')) {
            return;
        }

        // Global keyboard shortcuts
        if (e.altKey) {
            switch (e.key) {
                case 'h': // Alt+H for home/dashboard
                    e.preventDefault();
                    window.location.href = '/ui/';
                    break;
                case 'r': // Alt+R for runs
                    e.preventDefault();
                    window.location.href = '/ui/runs';
                    break;
                case 's': // Alt+S for search
                    e.preventDefault();
                    focusSearchField();
                    break;
                case 'm': // Alt+M for main content
                    e.preventDefault();
                    focusMainContent();
                    break;
            }
        }
    });
}

/**
 * Handle Tab navigation and focus trapping
 */
function handleTabNavigation(e) {
    const modal = document.querySelector('.modal[aria-hidden="false"]');
    if (modal) {
        trapFocusInModal(e, modal);
    }
}

/**
 * Trap focus within modal
 */
function trapFocusInModal(e, modal) {
    const focusableElements = modal.querySelectorAll(A11Y_CONFIG.focusableSelectors);
    const firstElement = focusableElements[0];
    const lastElement = focusableElements[focusableElements.length - 1];

    if (e.shiftKey) {
        // Shift + Tab
        if (document.activeElement === firstElement) {
            e.preventDefault();
            lastElement.focus();
        }
    } else {
        // Tab
        if (document.activeElement === lastElement) {
            e.preventDefault();
            firstElement.focus();
        }
    }
}

/**
 * Close topmost modal
 */
function closeTopModal() {
    const modal = document.querySelector('.modal[aria-hidden="false"]');
    if (modal) {
        closeModal(modal);
    }
}

/**
 * Close modal and restore focus
 */
function closeModal(modal) {
    modal.setAttribute('aria-hidden', 'true');

    // Restore focus to trigger element
    const triggerElement = modal.getAttribute('data-trigger-element');
    if (triggerElement) {
        const trigger = document.getElementById(triggerElement);
        if (trigger) {
            trigger.focus();
        }
    }

    announceToScreenReader('Dialog closed');
}

/**
 * Setup focus management
 */
function setupFocusManagement() {
    // Track focus for debugging
    if (window.TagSentinel && window.TagSentinel.config && window.TagSentinel.config.debug) {
        document.addEventListener('focusin', function(e) {
            console.log('Focus:', e.target);
        });
    }

    // Manage focus indicators
    document.addEventListener('mousedown', function() {
        document.body.classList.add('mouse-navigation');
        document.body.classList.remove('keyboard-navigation');
    });

    document.addEventListener('keydown', function(e) {
        if (e.key === 'Tab') {
            document.body.classList.add('keyboard-navigation');
            document.body.classList.remove('mouse-navigation');
        }
    });
}

/**
 * Setup live regions for screen reader announcements
 */
function setupLiveRegions() {
    // Ensure live region exists
    let liveRegion = document.getElementById(A11Y_CONFIG.liveRegionId);
    if (!liveRegion) {
        liveRegion = document.createElement('div');
        liveRegion.id = A11Y_CONFIG.liveRegionId;
        liveRegion.className = 'sr-only';
        liveRegion.setAttribute('aria-live', 'polite');
        liveRegion.setAttribute('aria-atomic', 'true');
        document.body.appendChild(liveRegion);
    }

    // Create urgent announcement region
    let urgentRegion = document.getElementById('sr-announcements-urgent');
    if (!urgentRegion) {
        urgentRegion = document.createElement('div');
        urgentRegion.id = 'sr-announcements-urgent';
        urgentRegion.className = 'sr-only';
        urgentRegion.setAttribute('aria-live', 'assertive');
        urgentRegion.setAttribute('aria-atomic', 'true');
        document.body.appendChild(urgentRegion);
    }
}

/**
 * Setup modal accessibility
 */
function setupModalAccessibility() {
    // Find all modal triggers
    const modalTriggers = document.querySelectorAll('[data-modal-target]');

    modalTriggers.forEach(trigger => {
        trigger.addEventListener('click', function(e) {
            e.preventDefault();
            const modalId = this.getAttribute('data-modal-target');
            const modal = document.getElementById(modalId);

            if (modal) {
                openModal(modal, this);
            }
        });
    });

    // Find all modal close buttons
    const closeButtons = document.querySelectorAll('[data-modal-close]');

    closeButtons.forEach(button => {
        button.addEventListener('click', function() {
            const modal = this.closest('.modal');
            if (modal) {
                closeModal(modal);
            }
        });
    });

    // Close modal on background click
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('modal')) {
            closeModal(e.target);
        }
    });
}

/**
 * Open modal with accessibility features
 */
function openModal(modal, triggerElement) {
    // Store trigger element for focus restoration
    if (triggerElement) {
        modal.setAttribute('data-trigger-element', triggerElement.id || '');
    }

    modal.setAttribute('aria-hidden', 'false');

    // Focus first focusable element in modal
    setTimeout(() => {
        const firstFocusable = modal.querySelector(A11Y_CONFIG.focusableSelectors);
        if (firstFocusable) {
            firstFocusable.focus();
        }
    }, 100);

    announceToScreenReader('Dialog opened');
}

/**
 * Setup table accessibility enhancements
 */
function setupTableAccessibility() {
    const tables = document.querySelectorAll('.table');

    tables.forEach(table => {
        // Add table navigation
        table.addEventListener('keydown', function(e) {
            if (e.target.matches('td, th')) {
                handleTableNavigation(e, table);
            }
        });

        // Ensure proper ARIA labels
        ensureTableAccessibility(table);
    });
}

/**
 * Handle table keyboard navigation
 */
function handleTableNavigation(e, table) {
    const cell = e.target;
    const row = cell.parentElement;
    const cellIndex = Array.from(row.children).indexOf(cell);
    const rowIndex = Array.from(table.querySelectorAll('tr')).indexOf(row);

    let targetCell = null;

    switch (e.key) {
        case 'ArrowUp':
            e.preventDefault();
            const prevRow = table.querySelectorAll('tr')[rowIndex - 1];
            if (prevRow) {
                targetCell = prevRow.children[cellIndex];
            }
            break;

        case 'ArrowDown':
            e.preventDefault();
            const nextRow = table.querySelectorAll('tr')[rowIndex + 1];
            if (nextRow) {
                targetCell = nextRow.children[cellIndex];
            }
            break;

        case 'ArrowLeft':
            e.preventDefault();
            targetCell = cell.previousElementSibling;
            break;

        case 'ArrowRight':
            e.preventDefault();
            targetCell = cell.nextElementSibling;
            break;

        case 'Home':
            e.preventDefault();
            targetCell = row.children[0];
            break;

        case 'End':
            e.preventDefault();
            targetCell = row.children[row.children.length - 1];
            break;
    }

    if (targetCell) {
        targetCell.focus();
        targetCell.scrollIntoView({ block: 'nearest' });
    }
}

/**
 * Ensure table has proper accessibility features
 */
function ensureTableAccessibility(table) {
    // Add caption if missing
    if (!table.querySelector('caption')) {
        const caption = document.createElement('caption');
        caption.className = 'sr-only';
        caption.textContent = 'Data table';
        table.prepend(caption);
    }

    // Ensure proper scope attributes
    const headers = table.querySelectorAll('th');
    headers.forEach(header => {
        if (!header.hasAttribute('scope')) {
            if (header.closest('thead')) {
                header.setAttribute('scope', 'col');
            } else {
                header.setAttribute('scope', 'row');
            }
        }
    });

    // Make cells focusable for keyboard navigation
    const cells = table.querySelectorAll('td, th');
    cells.forEach(cell => {
        if (!cell.hasAttribute('tabindex')) {
            cell.setAttribute('tabindex', '-1');
        }
    });

    // Make first cell focusable
    const firstCell = table.querySelector('td, th');
    if (firstCell) {
        firstCell.setAttribute('tabindex', '0');
    }
}

/**
 * Setup form accessibility enhancements
 */
function setupFormAccessibility() {
    const forms = document.querySelectorAll('form');

    forms.forEach(form => {
        // Ensure proper labeling
        ensureFormAccessibility(form);

        // Add form navigation
        form.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                // Clear form or close dialog
                const modal = form.closest('.modal');
                if (modal) {
                    closeModal(modal);
                }
            }
        });
    });
}

/**
 * Ensure form has proper accessibility features
 */
function ensureFormAccessibility(form) {
    // Associate labels with inputs
    const inputs = form.querySelectorAll('input, select, textarea');

    inputs.forEach(input => {
        // Ensure input has ID
        if (!input.id) {
            input.id = 'input-' + Math.random().toString(36).substr(2, 9);
        }

        // Find or create label
        let label = form.querySelector(`label[for="${input.id}"]`);
        if (!label) {
            label = input.closest('.form-group')?.querySelector('label');
            if (label && !label.hasAttribute('for')) {
                label.setAttribute('for', input.id);
            }
        }

        // Add aria-required for required fields
        if (input.hasAttribute('required') && !input.hasAttribute('aria-required')) {
            input.setAttribute('aria-required', 'true');
        }
    });

    // Ensure fieldsets have legends
    const fieldsets = form.querySelectorAll('fieldset');
    fieldsets.forEach(fieldset => {
        if (!fieldset.querySelector('legend')) {
            const legend = document.createElement('legend');
            legend.className = 'sr-only';
            legend.textContent = 'Form section';
            fieldset.prepend(legend);
        }
    });
}

/**
 * Setup tooltip accessibility
 */
function setupTooltipAccessibility() {
    const tooltips = document.querySelectorAll('.tooltip');

    tooltips.forEach(tooltip => {
        const trigger = tooltip;
        const text = tooltip.querySelector('.tooltip-text');

        if (!text) return;

        // Ensure proper ARIA attributes
        const textId = 'tooltip-' + Math.random().toString(36).substr(2, 9);
        text.id = textId;
        trigger.setAttribute('aria-describedby', textId);

        // Handle keyboard interaction
        trigger.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                trigger.blur();
            }
        });
    });
}

/**
 * Focus search field
 */
function focusSearchField() {
    const searchField = document.querySelector('input[type="search"], input[name="search"], #search');
    if (searchField) {
        searchField.focus();
        announceToScreenReader('Search field focused');
    }
}

/**
 * Focus main content
 */
function focusMainContent() {
    const mainContent = document.getElementById('main-content') || document.querySelector('main');
    if (mainContent) {
        mainContent.focus();
        announceToScreenReader('Main content focused');
    }
}

/**
 * Announce message to screen readers
 */
function announceToScreenReader(message, urgent = false) {
    const regionId = urgent ? 'sr-announcements-urgent' : A11Y_CONFIG.liveRegionId;
    const region = document.getElementById(regionId);

    if (region) {
        // Clear first, then announce
        region.textContent = '';

        setTimeout(() => {
            region.textContent = message;
        }, A11Y_CONFIG.announceDelay);

        // Clear after announcement
        setTimeout(() => {
            region.textContent = '';
        }, urgent ? 5000 : 3000);
    }
}

/**
 * Check if element is visible to screen readers
 */
function isVisibleToScreenReader(element) {
    const style = window.getComputedStyle(element);
    return style.display !== 'none' &&
           style.visibility !== 'hidden' &&
           style.opacity !== '0' &&
           !element.hasAttribute('aria-hidden') ||
           element.getAttribute('aria-hidden') !== 'true';
}

/**
 * Ensure element is accessible
 */
function ensureElementAccessibility(element) {
    // Check for missing alt text on images
    if (element.tagName === 'IMG' && !element.hasAttribute('alt')) {
        console.warn('Image missing alt text:', element);
        element.setAttribute('alt', '');
    }

    // Check for missing labels on form controls
    if (element.matches('input, select, textarea') && !element.hasAttribute('aria-label') && !element.hasAttribute('aria-labelledby')) {
        const label = document.querySelector(`label[for="${element.id}"]`);
        if (!label) {
            console.warn('Form control missing label:', element);
        }
    }

    // Check for missing headings
    if (element.matches('h1, h2, h3, h4, h5, h6')) {
        if (!element.textContent.trim()) {
            console.warn('Empty heading:', element);
        }
    }
}

// Export accessibility utilities
window.TagSentinel = window.TagSentinel || {};
window.TagSentinel.a11y = {
    announceToScreenReader,
    openModal,
    closeModal,
    focusMainContent,
    focusSearchField,
    ensureElementAccessibility,
    isVisibleToScreenReader
};