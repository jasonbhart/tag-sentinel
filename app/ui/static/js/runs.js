/**
 * Tag Sentinel UI - Runs Page JavaScript
 * Handles audit runs list, filtering, sorting, and pagination
 */

// Page state
let currentFilters = {};
let currentSort = { key: 'created_at', direction: 'desc' };
let currentCursor = null;
let nextCursor = null;
let prevCursor = null;
let cursorStack = [];
let pageSize = 20;
let totalRuns = 0;

// Initialize runs page
document.addEventListener('DOMContentLoaded', function() {
    initializeRunsPage();
});

/**
 * Initialize the runs page
 */
function initializeRunsPage() {
    setupEventListeners();
    setupFilterForm();
    loadRuns();

    console.log('Runs page initialized');
}

/**
 * Set up event listeners
 */
function setupEventListeners() {
    // Filter toggle
    const filterBtn = document.getElementById('filter-btn');
    const filterPanel = document.getElementById('filter-panel');

    if (filterBtn && filterPanel) {
        filterBtn.addEventListener('click', function() {
            toggleFilterPanel();
        });
    }

    // Refresh button
    const refreshBtn = document.getElementById('refresh-btn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', function() {
            loadRuns();
            window.TagSentinel.utils.announceToScreenReader('Refreshing audit runs');
        });
    }

    // Export button
    const exportBtn = document.getElementById('export-btn');
    if (exportBtn) {
        exportBtn.addEventListener('click', function() {
            exportRuns();
        });
    }

    // Retry button
    const retryBtn = document.getElementById('retry-btn');
    if (retryBtn) {
        retryBtn.addEventListener('click', function() {
            hideErrorState();
            loadRuns();
        });
    }

    // Clear filters buttons
    const clearFiltersBtn = document.getElementById('clear-filters');
    const clearFiltersEmptyBtn = document.getElementById('clear-filters-empty');

    if (clearFiltersBtn) {
        clearFiltersBtn.addEventListener('click', clearAllFilters);
    }

    if (clearFiltersEmptyBtn) {
        clearFiltersEmptyBtn.addEventListener('click', clearAllFilters);
    }

    // Pagination
    const prevBtn = document.getElementById('prev-page');
    const nextBtn = document.getElementById('next-page');

    if (prevBtn) {
        prevBtn.addEventListener('click', function() {
            if (cursorStack.length > 0) {
                currentCursor = cursorStack.pop();
                loadRuns();
            }
        });
    }

    if (nextBtn) {
        nextBtn.addEventListener('click', function() {
            if (nextCursor) {
                // Push current cursor so we can navigate back later
                cursorStack.push(currentCursor);
                currentCursor = nextCursor;
                loadRuns();
            }
        });
    }
}

/**
 * Set up filter form
 */
function setupFilterForm() {
    const filterForm = document.querySelector('.filter-form');
    if (!filterForm) return;

    // Form submission
    filterForm.addEventListener('submit', function(e) {
        e.preventDefault();
        applyFilters();
    });

    // Real-time search with debouncing
    const searchInput = document.getElementById('search-query');
    if (searchInput) {
        searchInput.addEventListener('input',
            window.TagSentinel.utils.debounce(function() {
                if (this.value.length === 0 || this.value.length >= 3) {
                    applyFilters();
                }
            }, 500)
        );
    }

    // Filter selects
    const filterSelects = filterForm.querySelectorAll('select');
    filterSelects.forEach(select => {
        select.addEventListener('change', applyFilters);
    });

    // Date inputs
    const dateInputs = filterForm.querySelectorAll('input[type="date"]');
    dateInputs.forEach(input => {
        input.addEventListener('change', applyFilters);
    });
}

/**
 * Toggle filter panel
 */
function toggleFilterPanel() {
    const filterBtn = document.getElementById('filter-btn');
    const filterPanel = document.getElementById('filter-panel');

    if (!filterBtn || !filterPanel) return;

    const isHidden = filterPanel.getAttribute('aria-hidden') === 'true';

    filterPanel.setAttribute('aria-hidden', !isHidden);
    filterBtn.setAttribute('aria-expanded', isHidden);

    if (isHidden) {
        filterPanel.style.display = 'block';
        // Focus first input
        const firstInput = filterPanel.querySelector('input, select');
        if (firstInput) {
            setTimeout(() => firstInput.focus(), 100);
        }
    } else {
        filterPanel.style.display = 'none';
    }
}

/**
 * Apply current filters
 */
function applyFilters() {
    // Collect filter values
    currentFilters = {
        search: document.getElementById('search-query')?.value || '',
        status: document.getElementById('status-filter')?.value || '',
        env: document.getElementById('environment-filter')?.value || '',
        date_from: document.getElementById('date-from')?.value || '',
        date_to: document.getElementById('date-to')?.value || ''
    };

    // Remove empty filters
    Object.keys(currentFilters).forEach(key => {
        if (!currentFilters[key]) {
            delete currentFilters[key];
        }
    });

    // Reset to first page
    currentCursor = null;
    nextCursor = null;
    prevCursor = null;
    cursorStack = [];

    // Load filtered results
    loadRuns();

    // Update URL with filters
    updateUrlWithFilters();

    // Announce filter application
    const filterCount = Object.keys(currentFilters).length;
    window.TagSentinel.utils.announceToScreenReader(
        `Filters applied. ${filterCount} filter${filterCount !== 1 ? 's' : ''} active.`
    );
}

/**
 * Clear all filters
 */
function clearAllFilters() {
    currentFilters = {};
    currentCursor = null;
    nextCursor = null;
    prevCursor = null;
    cursorStack = [];

    // Clear form inputs
    const filterForm = document.querySelector('.filter-form');
    if (filterForm) {
        filterForm.reset();
    }

    // Load unfiltered results
    loadRuns();

    // Update URL
    updateUrlWithFilters();

    // Close filter panel
    const filterPanel = document.getElementById('filter-panel');
    if (filterPanel) {
        filterPanel.setAttribute('aria-hidden', 'true');
        document.getElementById('filter-btn')?.setAttribute('aria-expanded', 'false');
    }

    window.TagSentinel.utils.announceToScreenReader('Filters cleared');
}

/**
 * Load runs with current filters and sorting
 */
async function loadRuns() {
    showLoadingState();

    try {
        const params = {
            ...currentFilters,
            limit: pageSize,
            sort_by: currentSort.key,
            sort_order: currentSort.direction
        };

        // Only add cursor if we have one
        if (currentCursor) {
            params.cursor = currentCursor;
        }

        const response = await window.TagSentinel.api.getRuns(params);

        if (Array.isArray(response?.audits)) {
            // Update cursor information
            nextCursor = response.next_cursor;
            prevCursor = response.prev_cursor;

            displayRuns(response);
            updatePagination(response);
            hideLoadingState();
        } else {
            showEmptyState();
        }

    } catch (error) {
        console.error('Failed to load runs:', error);
        showErrorState(error.message);
    }
}

/**
 * Display runs in table
 */
function displayRuns(response) {
    const tbody = document.getElementById('runs-tbody');
    if (!tbody) return;

    const runs = response.audits || [];
    totalRuns = response.total_count || 0;

    // Update results count
    updateResultsCount(runs.length, totalRuns);

    if (runs.length === 0) {
        showEmptyState();
        return;
    }

    // Generate table rows
    tbody.innerHTML = runs.map(run => createRunRow(run)).join('');

    // Show results table
    const resultsTable = document.getElementById('results-table');
    if (resultsTable) {
        resultsTable.style.display = 'block';
    }

    // Hide empty/error states
    hideEmptyState();
    hideErrorState();
}

/**
 * Create table row for a run
 */
function createRunRow(run) {
    const statusClass = getStatusClass(run.status);
    const statusText = (run.status || 'unknown').charAt(0).toUpperCase() + (run.status || 'unknown').slice(1);

    return `
        <tr data-run-id="${escapeHtml(run.id)}">
            <td>
                <a href="/ui/runs/${escapeHtml(run.id)}" class="run-link">
                    ${escapeHtml(run.site_id || 'Unknown')}
                </a>
            </td>
            <td>
                <span class="environment-tag">${escapeHtml(run.env || 'production')}</span>
            </td>
            <td>
                <span class="badge badge-${statusClass}" title="${statusText}">
                    ${statusText}
                </span>
                ${run.status === 'running' ? createProgressBar(run) : ''}
            </td>
            <td>
                <time datetime="${run.started_at || run.created_at}" title="${formatFullDate(run.started_at || run.created_at)}">
                    ${formatRelativeDate(run.started_at || run.created_at)}
                </time>
            </td>
            <td>
                ${run.finished_at ? `
                    <time datetime="${run.finished_at}" title="${formatFullDate(run.finished_at)}">
                        ${formatRelativeDate(run.finished_at)}
                    </time>
                ` : '<span class="text-muted">—</span>'}
            </td>
            <td>
                <span class="pages-count">${run.summary?.pages_processed ?? 0}</span>
            </td>
            <td>
                <span class="issues-count ${(run.summary?.pages_failed ?? 0) > 0 ? 'has-issues' : ''}">
                    ${run.summary?.pages_failed ?? 0}
                </span>
            </td>
            <td>
                <div class="action-buttons">
                    <a href="/ui/runs/${escapeHtml(run.id)}"
                       class="btn btn-secondary btn-sm"
                       title="View run details">
                        View
                    </a>
                    ${run.status === 'completed' ? `
                        <button class="btn btn-secondary btn-sm"
                                onclick="exportRun('${escapeHtml(run.id)}')"
                                title="Export run data">
                            Export
                        </button>
                    ` : ''}
                </div>
            </td>
        </tr>
    `;
}

/**
 * Create progress bar for running audits
 */
function createProgressBar(run) {
    const progress = run.progress_percent || 0;
    return `
        <div class="progress-cell">
            <div class="progress-bar-mini" role="progressbar"
                 aria-valuenow="${progress}"
                 aria-valuemin="0"
                 aria-valuemax="100"
                 aria-label="Audit progress">
                <div class="progress-fill" style="width: ${progress}%"></div>
            </div>
            <span class="progress-text">${progress}%</span>
        </div>
    `;
}

/**
 * Update pagination controls
 */
function updatePagination(response) {
    const runs = response.audits || [];

    // Update pagination info
    const paginationInfo = document.getElementById('pagination-info');
    if (paginationInfo) {
        if (totalRuns > 0) {
            paginationInfo.textContent = `Showing ${runs.length} of ${totalRuns} runs`;
        } else {
            paginationInfo.textContent = `Showing ${runs.length} runs`;
        }
    }

    // Update pagination buttons
    const prevBtn = document.getElementById('prev-page');
    const nextBtn = document.getElementById('next-page');

    if (prevBtn) {
        prevBtn.disabled = !prevCursor;
    }

    if (nextBtn) {
        nextBtn.disabled = !nextCursor;
    }

    // Hide page numbers for cursor-based pagination
    const pagesContainer = document.getElementById('pagination-pages');
    if (pagesContainer) {
        pagesContainer.style.display = 'none';
    }
}


/**
 * Update results count display
 */
function updateResultsCount(displayCount, totalCount) {
    const resultsCount = document.getElementById('results-count');
    if (resultsCount) {
        resultsCount.textContent = `Showing ${displayCount} of ${totalCount} runs`;
    }
}

/**
 * Show loading state
 */
function showLoadingState() {
    const loadingState = document.getElementById('loading-state');
    const resultsTable = document.getElementById('results-table');

    if (loadingState) loadingState.style.display = 'block';
    if (resultsTable) resultsTable.style.display = 'none';

    hideEmptyState();
    hideErrorState();
}

/**
 * Hide loading state
 */
function hideLoadingState() {
    const loadingState = document.getElementById('loading-state');
    if (loadingState) {
        loadingState.style.display = 'none';
    }
}

/**
 * Show empty state
 */
function showEmptyState() {
    const emptyState = document.getElementById('empty-state');
    const resultsTable = document.getElementById('results-table');

    if (emptyState) emptyState.style.display = 'block';
    if (resultsTable) resultsTable.style.display = 'none';

    hideLoadingState();
    hideErrorState();
}

/**
 * Hide empty state
 */
function hideEmptyState() {
    const emptyState = document.getElementById('empty-state');
    if (emptyState) {
        emptyState.style.display = 'none';
    }
}

/**
 * Show error state
 */
function showErrorState(message) {
    const errorState = document.getElementById('error-state');
    const errorMessage = document.getElementById('error-message');
    const resultsTable = document.getElementById('results-table');

    if (errorState) errorState.style.display = 'block';
    if (resultsTable) resultsTable.style.display = 'none';

    if (errorMessage && message) {
        errorMessage.textContent = message;
    }

    hideLoadingState();
    hideEmptyState();
}

/**
 * Hide error state
 */
function hideErrorState() {
    const errorState = document.getElementById('error-state');
    if (errorState) {
        errorState.style.display = 'none';
    }
}

/**
 * Update URL with current filters
 */
function updateUrlWithFilters() {
    const url = new URL(window.location);

    // Clear existing filter params
    url.searchParams.delete('search');
    url.searchParams.delete('status');
    url.searchParams.delete('env');
    url.searchParams.delete('environment'); // Also clear legacy param
    url.searchParams.delete('date_from');
    url.searchParams.delete('date_to');
    url.searchParams.delete('page');

    // Add current filters
    Object.keys(currentFilters).forEach(key => {
        if (currentFilters[key]) {
            url.searchParams.set(key, currentFilters[key]);
        }
    });

    if (currentCursor) {
        url.searchParams.set('cursor', currentCursor);
    }

    // Update URL without page reload
    window.history.replaceState({}, '', url);
}

/**
 * Export current runs
 */
async function exportRuns() {
    try {
        // Note: This function exports all runs matching current filters
        // Since the API only supports per-audit exports, we need to implement
        // a different approach or notify the user that bulk export is not available
        console.warn('Bulk runs export not supported by current API. Use individual run exports.');
        window.TagSentinel.utils.announceToScreenReader('Bulk export not supported. Use individual run exports.');

    } catch (error) {
        console.error('Export failed:', error);
        window.TagSentinel.utils.announceToScreenReader('Export failed');
    }
}

/**
 * Export single run
 */
async function exportRun(runId) {
    try {
        const exportUrl = `${window.TagSentinel.config.apiBaseUrl}/exports/${runId}/requests.csv`;

        // Fetch the export data with proper error handling
        const response = await fetch(exportUrl, { credentials: 'include' });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        // Get the response as a blob for download
        const blob = await response.blob();

        // Create download link only after successful fetch
        const link = document.createElement('a');
        const blobUrl = URL.createObjectURL(blob);
        link.href = blobUrl;
        link.download = `run-${runId}-requests.csv`;
        link.style.display = 'none';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        // Clean up blob URL after a short delay to ensure download starts
        setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);

        window.TagSentinel.utils.announceToScreenReader('Run export completed successfully');

    } catch (error) {
        console.error('Run export failed:', error);
        window.TagSentinel.utils.announceToScreenReader(`Run export failed: ${error.message}`);
    }
}

/**
 * Get status CSS class
 */
function getStatusClass(status) {
    const statusMap = {
        'completed': 'success',
        'running': 'primary',
        'failed': 'danger',
        'queued': 'secondary'
    };
    return statusMap[status] || 'secondary';
}

/**
 * Format relative date
 */
function formatRelativeDate(dateString) {
    if (!dateString) return 'Unknown';

    const date = new Date(dateString);
    const now = new Date();
    const diff = now.getTime() - date.getTime();

    const minutes = Math.floor(diff / (1000 * 60));
    const hours = Math.floor(diff / (1000 * 60 * 60));
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));

    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    if (days < 7) return `${days}d ago`;

    return date.toLocaleDateString();
}

/**
 * Format full date
 */
function formatFullDate(dateString) {
    if (!dateString) return 'Unknown';
    return new Date(dateString).toLocaleString();
}

/**
 * Escape HTML
 */
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
