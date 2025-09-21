/**
 * Tag Sentinel UI - Run Detail JavaScript
 * Handles run detail page with tabbed interface and data loading
 */

// Page state
let runId = null;
let currentTab = 'pages';
let tabData = {};
let tabFilters = {};
let auditDetailCache = null;

// Initialize run detail page
document.addEventListener('DOMContentLoaded', function() {
    initializeRunDetailPage();
});

/**
 * Initialize the run detail page
 */
function initializeRunDetailPage() {
    // Extract run ID from URL
    const pathParts = window.location.pathname.split('/');
    runId = pathParts[pathParts.indexOf('runs') + 1];

    if (!runId) {
        showError('Invalid run ID');
        return;
    }

    setupEventListeners();
    setupTabNavigation();
    loadRunOverview();
    loadTabData(currentTab);

    console.log('Run detail page initialized for run:', runId);
}

/**
 * Set up event listeners
 */
function setupEventListeners() {
    // Refresh button
    const refreshBtn = document.getElementById('refresh-btn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', function() {
            refreshAllData();
            window.TagSentinel.utils.announceToScreenReader('Refreshing run data');
        });
    }

    // Export button
    const exportBtn = document.getElementById('export-run-btn');
    if (exportBtn) {
        exportBtn.addEventListener('click', function() {
            exportRunData();
        });
    }

    // Filter inputs for each tab
    setupTabFilters();
}

/**
 * Set up tab navigation
 */
function setupTabNavigation() {
    const tabButtons = document.querySelectorAll('.tab-button');

    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            const tabName = this.getAttribute('data-tab');
            switchTab(tabName);
        });

        // Keyboard navigation
        button.addEventListener('keydown', function(e) {
            if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
                e.preventDefault();
                navigateTabs(e.key === 'ArrowRight');
            }
        });
    });
}

/**
 * Set up tab filters
 */
function setupTabFilters() {
    // Pages filters
    const pagesStatusFilter = document.getElementById('pages-status-filter');
    const pagesSearch = document.getElementById('pages-search');

    if (pagesStatusFilter) {
        pagesStatusFilter.addEventListener('change', () => applyTabFilter('pages'));
    }

    if (pagesSearch) {
        pagesSearch.addEventListener('input',
            window.TagSentinel.utils.debounce(() => applyTabFilter('pages'), 300)
        );
    }

    // Tags filters
    const tagsVendorFilter = document.getElementById('tags-vendor-filter');
    const tagsStatusFilter = document.getElementById('tags-status-filter');

    if (tagsVendorFilter) {
        tagsVendorFilter.addEventListener('change', () => applyTabFilter('tags'));
    }

    if (tagsStatusFilter) {
        tagsStatusFilter.addEventListener('change', () => applyTabFilter('tags'));
    }

    // Similar setups for other tab filters...
    setupOtherTabFilters();
}

/**
 * Set up filters for other tabs
 */
function setupOtherTabFilters() {
    // Variables filters
    const variablesTypeFilter = document.getElementById('variables-type-filter');
    if (variablesTypeFilter) {
        variablesTypeFilter.addEventListener('change', () => applyTabFilter('variables'));
    }

    // Cookies filters
    const cookiesCategoryFilter = document.getElementById('cookies-category-filter');
    if (cookiesCategoryFilter) {
        cookiesCategoryFilter.addEventListener('change', () => applyTabFilter('cookies'));
    }

    // Rules filters
    const rulesSeverityFilter = document.getElementById('rules-severity-filter');
    if (rulesSeverityFilter) {
        rulesSeverityFilter.addEventListener('change', () => applyTabFilter('rules'));
    }
}

/**
 * Switch to a different tab
 */
function switchTab(tabName) {
    // Update active tab button
    document.querySelectorAll('.tab-button').forEach(btn => {
        const isActive = btn.getAttribute('data-tab') === tabName;
        btn.classList.toggle('active', isActive);
        btn.setAttribute('aria-selected', isActive);
    });

    // Update active tab panel
    document.querySelectorAll('.tab-panel').forEach(panel => {
        const isActive = panel.id === `${tabName}-panel`;
        panel.classList.toggle('active', isActive);
        panel.setAttribute('aria-hidden', !isActive);
    });

    currentTab = tabName;

    // Load tab data if not already loaded
    if (!tabData[tabName]) {
        loadTabData(tabName);
    }

    // Announce tab change
    window.TagSentinel.utils.announceToScreenReader(`Switched to ${tabName} tab`);
}

/**
 * Navigate tabs with keyboard
 */
function navigateTabs(forward) {
    const tabs = Array.from(document.querySelectorAll('.tab-button'));
    const currentIndex = tabs.findIndex(tab => tab.getAttribute('data-tab') === currentTab);

    let newIndex;
    if (forward) {
        newIndex = currentIndex < tabs.length - 1 ? currentIndex + 1 : 0;
    } else {
        newIndex = currentIndex > 0 ? currentIndex - 1 : tabs.length - 1;
    }

    const newTab = tabs[newIndex].getAttribute('data-tab');
    switchTab(newTab);
    tabs[newIndex].focus();
}

/**
 * Load run overview data
 */
async function loadRunOverview() {
    try {
        const response = await getCachedAuditDetail();
        displayRunOverview(response);

    } catch (error) {
        console.error('Failed to load run overview:', error);
        showOverviewError(error.message);
    }
}

/**
 * Refresh run data (clears cache and reloads)
 */
async function refreshRun() {
    clearAuditDetailCache();
    await loadRunOverview();
    await loadTabData(currentTab);
}

/**
 * Display run overview
 */
function displayRunOverview(data) {
    const overviewContainer = document.getElementById('run-overview');
    if (!overviewContainer) return;

    const status = data.status || 'unknown';
    const statusClass = getStatusClass(status);

    // Calculate duration from timestamps
    let duration = 0;
    if (data.finished_at && data.started_at) {
        duration = new Date(data.finished_at) - new Date(data.started_at);
    } else if (data.summary?.duration_seconds) {
        duration = data.summary.duration_seconds * 1000;
    }

    overviewContainer.innerHTML = `
        <div class="overview-item">
            <span class="overview-label">Status</span>
            <div class="overview-value">
                <span class="badge badge-${statusClass}">${status.charAt(0).toUpperCase() + status.slice(1)}</span>
            </div>
        </div>
        <div class="overview-item">
            <span class="overview-label">Site</span>
            <div class="overview-value">${escapeHtml(data.site_id || 'Unknown')}</div>
        </div>
        <div class="overview-item">
            <span class="overview-label">Environment</span>
            <div class="overview-value">${escapeHtml(data.env || 'production')}</div>
        </div>
        <div class="overview-item">
            <span class="overview-label">Started</span>
            <div class="overview-value">
                <time datetime="${data.started_at || data.created_at}" title="${formatFullDate(data.started_at || data.created_at)}">
                    ${formatRelativeDate(data.started_at || data.created_at)}
                </time>
            </div>
        </div>
        <div class="overview-item">
            <span class="overview-label">Duration</span>
            <div class="overview-value">${formatDuration(duration)}</div>
        </div>
        <div class="overview-item">
            <span class="overview-label">Pages</span>
            <div class="overview-value">${data.summary?.pages_processed || 0}</div>
        </div>
        <div class="overview-item">
            <span class="overview-label">Issues</span>
            <div class="overview-value ${(data.summary?.pages_failed || 0) > 0 ? 'text-danger' : 'text-success'}">
                ${data.summary?.pages_failed || 0}
            </div>
        </div>
        <div class="overview-item">
            <span class="overview-label">Tags Found</span>
            <div class="overview-value">${data.summary?.tags_detected || 0}</div>
        </div>
    `;

    // Update page title with site info
    if (data.site_id) {
        document.title = `${data.site_id} - Run ${runId} - Tag Sentinel`;
    }
}

/**
 * Show overview error
 */
function showOverviewError(message) {
    const overviewContainer = document.getElementById('run-overview');
    if (overviewContainer) {
        overviewContainer.innerHTML = `
            <div class="error-message">
                <p>Unable to load run overview: ${escapeHtml(message)}</p>
                <button onclick="loadRunOverview()" class="btn btn-primary btn-sm">Retry</button>
            </div>
        `;
    }
}

/**
 * Load tab data
 */
async function loadTabData(tabName) {
    try {
        showTabLoading(tabName);

        let data;
        switch (tabName) {
            case 'pages':
                data = await loadPagesData();
                break;
            case 'tags':
                data = await loadTagsData();
                break;
            case 'health':
                data = await loadHealthData();
                break;
            case 'duplicates':
                data = await loadDuplicatesData();
                break;
            case 'variables':
                data = await loadVariablesData();
                break;
            case 'cookies':
                data = await loadCookiesData();
                break;
            case 'rules':
                data = await loadRulesData();
                break;
            default:
                throw new Error(`Unknown tab: ${tabName}`);
        }

        tabData[tabName] = data;
        displayTabData(tabName, data);
        updateTabCount(tabName, extractDataCount(data));

    } catch (error) {
        console.error(`Failed to load ${tabName} data:`, error);
        showTabError(tabName, error.message);
    }
}

/**
 * Get cached audit detail or fetch if not cached
 */
async function getCachedAuditDetail() {
    if (!auditDetailCache) {
        try {
            auditDetailCache = await window.TagSentinel.api.getAuditDetail(runId);
        } catch (error) {
            console.error('Failed to load audit detail:', error);
            auditDetailCache = null; // Reset cache to allow retries
            throw error; // Rethrow to let callers handle the error
        }
    }
    return auditDetailCache;
}

/**
 * Clear audit detail cache
 */
function clearAuditDetailCache() {
    auditDetailCache = null;
}

/**
 * Load pages data
 */
async function loadPagesData() {
    try {
        const response = await getCachedAuditDetail();
        return { items: response.pages || [] };
    } catch (error) {
        console.error('Failed to load pages data:', error);
        throw error; // Rethrow to let loadTabData handle the error state
    }
}

/**
 * Load tags data
 */
async function loadTagsData() {
    try {
        const response = await getCachedAuditDetail();
        return { items: response.tags || [] };
    } catch (error) {
        console.error('Failed to load tags data:', error);
        throw error; // Rethrow to let loadTabData handle the error state
    }
}

/**
 * Load health data
 */
async function loadHealthData() {
    try {
        const response = await getCachedAuditDetail();
        return {
            load_performance: response.health?.load_performance || "Unknown",
            error_rate: response.health?.error_rate || "0%",
            tag_coverage: response.health?.tag_coverage || "0%",
            issues: response.health?.issues || []
        };
    } catch (error) {
        console.error('Failed to load health data:', error);
        throw error; // Rethrow to let loadTabData handle the error state
    }
}

/**
 * Load duplicates data
 */
async function loadDuplicatesData() {
    try {
        const response = await getCachedAuditDetail();
        return { items: response.duplicates || [] };
    } catch (error) {
        console.error('Failed to load duplicates data:', error);
        throw error; // Rethrow to let loadTabData handle the error state
    }
}

/**
 * Load variables data
 */
async function loadVariablesData() {
    try {
        const response = await getCachedAuditDetail();
        return { items: response.variables || [] };
    } catch (error) {
        console.error('Failed to load variables data:', error);
        throw error; // Rethrow to let loadTabData handle the error state
    }
}

/**
 * Load cookies data
 */
async function loadCookiesData() {
    try {
        const response = await getCachedAuditDetail();
        return {
            items: response.cookies || [],
            privacy_summary: response.privacy_summary || {}
        };
    } catch (error) {
        console.error('Failed to load cookies data:', error);
        throw error; // Rethrow to let loadTabData handle the error state
    }
}

/**
 * Load rules data
 */
async function loadRulesData() {
    try {
        const response = await getCachedAuditDetail();
        return { items: response.rules || [] };
    } catch (error) {
        console.error('Failed to load rules data:', error);
        throw error; // Rethrow to let loadTabData handle the error state
    }
}

/**
 * Display tab data
 */
function displayTabData(tabName, data) {
    switch (tabName) {
        case 'pages':
            displayPagesData(data);
            break;
        case 'tags':
            displayTagsData(data);
            break;
        case 'health':
            displayHealthData(data);
            break;
        case 'duplicates':
            displayDuplicatesData(data);
            break;
        case 'variables':
            displayVariablesData(data);
            break;
        case 'cookies':
            displayCookiesData(data);
            break;
        case 'rules':
            displayRulesData(data);
            break;
    }
}

/**
 * Display pages data
 */
function displayPagesData(data) {
    const tbody = document.getElementById('pages-tbody');
    if (!tbody || !data.items) return;

    tbody.innerHTML = data.items.map(page => `
        <tr>
            <td>
                <a href="/ui/runs/${runId}/pages/${page.id}" class="page-link">
                    ${escapeHtml(page.url)}
                </a>
            </td>
            <td>
                <span class="badge badge-${getStatusClass(page.status)}">
                    ${escapeHtml(page.status || 'unknown')}
                </span>
            </td>
            <td>${page.tags_count || 0}</td>
            <td>
                <span class="${page.issues_count > 0 ? 'text-danger' : 'text-success'}">
                    ${page.issues_count || 0}
                </span>
            </td>
            <td>${formatLoadTime(page.load_time)}</td>
            <td>
                <a href="/ui/runs/${runId}/pages/${page.id}" class="btn btn-secondary btn-sm">
                    Details
                </a>
            </td>
        </tr>
    `).join('');
}

/**
 * Display tags data
 */
function displayTagsData(data) {
    const tbody = document.getElementById('tags-tbody');
    if (!tbody || !data.items) return;

    tbody.innerHTML = data.items.map(tag => `
        <tr>
            <td>
                <span class="vendor-badge vendor-${tag.vendor}">
                    ${escapeHtml(tag.vendor.toUpperCase())}
                </span>
            </td>
            <td>${escapeHtml(tag.tag_name || 'Unknown')}</td>
            <td>${escapeHtml(tag.measurement_id || '—')}</td>
            <td>${tag.pages_detected || 0}</td>
            <td>${tag.total_events || 0}</td>
            <td>
                <span class="badge badge-${getStatusClass(tag.status)}">
                    ${escapeHtml(tag.status || 'unknown')}
                </span>
            </td>
            <td>
                <button class="btn btn-secondary btn-sm" onclick="viewTagDetails('${tag.id}')">
                    Details
                </button>
            </td>
        </tr>
    `).join('');
}

/**
 * Display health data
 */
function displayHealthData(data) {
    // Update metrics
    updateMetric('load-performance', data.load_performance);
    updateMetric('error-rate', data.error_rate);
    updateMetric('tag-coverage', data.tag_coverage);

    // Update health issues table
    const tbody = document.getElementById('health-tbody');
    if (tbody && data.issues) {
        tbody.innerHTML = data.issues.map(issue => `
            <tr>
                <td>${escapeHtml(issue.type)}</td>
                <td>${issue.affected_tags || 0}</td>
                <td>${issue.affected_pages || 0}</td>
                <td>
                    <span class="badge badge-${getSeverityClass(issue.severity)}">
                        ${escapeHtml(issue.severity)}
                    </span>
                </td>
                <td>
                    <button class="btn btn-secondary btn-sm" onclick="viewHealthIssue('${issue.id}')">
                        View
                    </button>
                </td>
            </tr>
        `).join('');
    }
}

/**
 * Display duplicates data
 */
function displayDuplicatesData(data) {
    const tbody = document.getElementById('duplicates-tbody');
    if (!tbody || !data.items) return;

    tbody.innerHTML = data.items.map(duplicate => `
        <tr>
            <td>${escapeHtml(duplicate.event_type)}</td>
            <td>
                <a href="/ui/runs/${runId}/pages/${duplicate.page_id}">
                    ${escapeHtml(duplicate.page_url)}
                </a>
            </td>
            <td>
                <span class="occurrence-count ${duplicate.occurrences > 2 ? 'text-danger' : 'text-warning'}">
                    ${duplicate.occurrences}
                </span>
            </td>
            <td>${duplicate.tags_involved ? duplicate.tags_involved.join(', ') : '—'}</td>
            <td>
                <button class="btn btn-secondary btn-sm" onclick="viewDuplicateDetails('${duplicate.id}')">
                    Details
                </button>
            </td>
        </tr>
    `).join('');
}

/**
 * Display variables data
 */
function displayVariablesData(data) {
    const tbody = document.getElementById('variables-tbody');
    if (!tbody || !data.items) return;

    tbody.innerHTML = data.items.map(variable => `
        <tr>
            <td>${escapeHtml(variable.name)}</td>
            <td>
                <span class="variable-type type-${variable.type}">
                    ${escapeHtml(variable.type)}
                </span>
            </td>
            <td>${variable.pages_present || 0}/${variable.total_pages || 0}</td>
            <td>
                <span class="badge badge-${getValidationClass(variable.validation)}">
                    ${escapeHtml(variable.validation || 'unknown')}
                </span>
            </td>
            <td>
                <button class="btn btn-secondary btn-sm" onclick="viewVariableDetails('${variable.name}')">
                    Details
                </button>
            </td>
        </tr>
    `).join('');
}

/**
 * Display cookies data
 */
function displayCookiesData(data) {
    // Update privacy summary
    if (data.privacy_summary) {
        updatePrivacySummary(data.privacy_summary);
    }

    const tbody = document.getElementById('cookies-tbody');
    if (!tbody || !data.items) return;

    tbody.innerHTML = data.items.map(cookie => `
        <tr>
            <td>${escapeHtml(cookie.name)}</td>
            <td>${escapeHtml(cookie.domain)}</td>
            <td>
                <span class="category-badge category-${cookie.category}">
                    ${escapeHtml(cookie.category)}
                </span>
            </td>
            <td>${formatDuration(cookie.duration)}</td>
            <td>
                <span class="privacy-impact impact-${cookie.privacy_impact}">
                    ${escapeHtml(cookie.privacy_impact)}
                </span>
            </td>
            <td>
                <button class="btn btn-secondary btn-sm" onclick="viewCookieDetails('${cookie.name}')">
                    Details
                </button>
            </td>
        </tr>
    `).join('');
}

/**
 * Display rules data
 */
function displayRulesData(data) {
    const tbody = document.getElementById('rules-tbody');
    if (!tbody || !data.items) return;

    tbody.innerHTML = data.items.map(rule => `
        <tr>
            <td>${escapeHtml(rule.rule_name)}</td>
            <td>${escapeHtml(rule.description)}</td>
            <td>
                <span class="badge badge-${getSeverityClass(rule.severity)}">
                    ${escapeHtml(rule.severity)}
                </span>
            </td>
            <td>${rule.affected_pages || 0}</td>
            <td>
                <button class="btn btn-secondary btn-sm" onclick="viewRuleDetails('${rule.id}')">
                    Details
                </button>
            </td>
        </tr>
    `).join('');
}

/**
 * Update metric display
 */
function updateMetric(metricId, value) {
    const element = document.getElementById(metricId);
    if (element && value !== undefined) {
        element.textContent = value;
    }
}

/**
 * Update privacy summary
 */
function updatePrivacySummary(summary) {
    const complianceScore = document.getElementById('compliance-score');
    const thirdPartyCount = document.getElementById('third-party-count');

    if (complianceScore && summary.compliance_score !== undefined) {
        complianceScore.textContent = `${summary.compliance_score}%`;
        complianceScore.className = `compliance-score ${getComplianceClass(summary.compliance_score)}`;
    }

    if (thirdPartyCount && summary.third_party_count !== undefined) {
        thirdPartyCount.textContent = summary.third_party_count;
    }
}

/**
 * Update tab count
 */
function updateTabCount(tabName, count) {
    const countElement = document.getElementById(`${tabName}-count`);
    if (countElement) {
        countElement.textContent = count;
    }
}

/**
 * Show tab loading state
 */
function showTabLoading(tabName) {
    const tbody = document.getElementById(`${tabName}-tbody`);
    if (tbody) {
        tbody.innerHTML = `
            <tr>
                <td colspan="6" class="loading-container">
                    <div class="spinner" aria-hidden="true"></div>
                    <span>Loading ${tabName} data...</span>
                </td>
            </tr>
        `;
    }
}

/**
 * Show tab error
 */
function showTabError(tabName, message) {
    const tbody = document.getElementById(`${tabName}-tbody`);
    if (tbody) {
        tbody.innerHTML = `
            <tr>
                <td colspan="6" class="empty-state">
                    <div class="empty-state-icon">⚠️</div>
                    <h3 class="empty-state-title">Unable to load ${tabName}</h3>
                    <p class="empty-state-text">${escapeHtml(message)}</p>
                    <button onclick="loadTabData('${tabName}')" class="btn btn-primary">Retry</button>
                </td>
            </tr>
        `;
    }
}

/**
 * Apply tab filter
 */
function applyTabFilter(tabName) {
    // This would filter the displayed data based on current filter values
    // Implementation depends on specific filtering requirements
    console.log(`Applying filter for ${tabName} tab`);
}

/**
 * Refresh all data
 */
function refreshAllData() {
    // Clear cached data
    tabData = {};
    clearAuditDetailCache();

    // Reload overview and current tab
    loadRunOverview();
    loadTabData(currentTab);
}

/**
 * Export run data
 */
async function exportRunData() {
    try {
        // Define available exports based on actual API endpoints
        const exports = [
            { path: 'requests.json', name: 'Requests' },
            { path: 'cookies.csv', name: 'Cookies' },
            { path: 'tags.json', name: 'Tags' },
            { path: 'data-layer.json', name: 'Data Layer' }
        ];

        window.TagSentinel.utils.announceToScreenReader('Starting exports...');

        let successCount = 0;
        let failureCount = 0;

        // Download each export file using fetch for proper error handling
        for (const exportItem of exports) {
            try {
                const exportUrl = `${window.TagSentinel.config.apiBaseUrl}/exports/${runId}/${exportItem.path}`;

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
                link.download = `${runId}-${exportItem.path}`;
                link.style.display = 'none';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);

                // Clean up blob URL after a short delay to ensure download starts
                setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);

                successCount++;
                console.log(`Successfully exported ${exportItem.name}`);

                // Small delay between downloads to avoid overwhelming browser
                await new Promise(resolve => setTimeout(resolve, 100));

            } catch (exportError) {
                failureCount++;
                console.error(`Failed to export ${exportItem.name}:`, exportError);
                window.TagSentinel.utils.announceToScreenReader(`Failed to export ${exportItem.name}: ${exportError.message}`);
                // Continue with other exports even if one fails
            }
        }

        // Announce final results
        if (successCount > 0 && failureCount === 0) {
            window.TagSentinel.utils.announceToScreenReader(`All ${successCount} exports completed successfully`);
        } else if (successCount > 0 && failureCount > 0) {
            window.TagSentinel.utils.announceToScreenReader(`${successCount} exports succeeded, ${failureCount} failed`);
        } else {
            window.TagSentinel.utils.announceToScreenReader('All exports failed');
        }

    } catch (error) {
        console.error('Export failed:', error);
        window.TagSentinel.utils.announceToScreenReader('Export failed');
    }
}

/**
 * Extract count from various data structure formats
 */
function extractDataCount(data) {
    // Try different possible count fields based on known schema patterns
    if (Array.isArray(data?.items)) return data.items.length;
    if (Array.isArray(data?.cookies)) return data.cookies.length;
    if (Array.isArray(data?.audits)) return data.audits.length;
    if (Array.isArray(data?.rules)) return data.rules.length;
    if (Array.isArray(data?.tags)) return data.tags.length;
    if (Array.isArray(data?.issues)) return data.issues.length;
    if (typeof data?.total_count === 'number') return data.total_count;
    if (typeof data?.total === 'number') return data.total;
    if (Array.isArray(data)) return data.length;

    return 0;
}

/**
 * Helper functions
 */
function getStatusClass(status) {
    const statusMap = {
        'completed': 'success', 'success': 'success',
        'completed_with_issues': 'warning',
        'running': 'primary', 'in_progress': 'primary',
        'failed': 'danger', 'error': 'danger',
        'queued': 'secondary', 'pending': 'secondary',
        'warning': 'warning'
    };
    return statusMap[status] || 'secondary';
}

function getSeverityClass(severity) {
    const severityMap = {
        'critical': 'danger',
        'high': 'danger',
        'medium': 'warning',
        'low': 'secondary'
    };
    return severityMap[severity] || 'secondary';
}

function getValidationClass(validation) {
    const validationMap = {
        'valid': 'success',
        'invalid': 'danger',
        'warning': 'warning',
        'unknown': 'secondary'
    };
    return validationMap[validation] || 'secondary';
}

function getComplianceClass(score) {
    if (score >= 90) return 'text-success';
    if (score >= 70) return 'text-warning';
    return 'text-danger';
}

function formatLoadTime(milliseconds) {
    if (!milliseconds) return '—';
    if (milliseconds < 1000) return `${milliseconds}ms`;
    return `${(milliseconds / 1000).toFixed(1)}s`;
}

function formatDuration(milliseconds) {
    if (!milliseconds) return '—';
    const seconds = Math.floor(milliseconds / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (hours > 0) return `${hours}h ${minutes % 60}m`;
    if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
    return `${seconds}s`;
}

function formatRelativeDate(dateString) {
    if (!dateString) return 'Unknown';
    // Use the utility function from main.js
    return window.TagSentinel.utils.formatDate(dateString);
}

function formatFullDate(dateString) {
    if (!dateString) return 'Unknown';
    return new Date(dateString).toLocaleString();
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showError(message) {
    console.error(message);
    window.TagSentinel.utils.announceToScreenReader(`Error: ${message}`);
}

// Detail view functions (called by buttons)
window.viewTagDetails = function(tagId) {
    console.log('View tag details:', tagId);
    // Implementation would show tag detail modal or navigate to detail page
};

window.viewHealthIssue = function(issueId) {
    console.log('View health issue:', issueId);
    // Implementation would show issue detail modal
};

window.viewDuplicateDetails = function(duplicateId) {
    console.log('View duplicate details:', duplicateId);
    // Implementation would show duplicate detail modal
};

window.viewVariableDetails = function(variableName) {
    console.log('View variable details:', variableName);
    // Implementation would show variable detail modal
};

window.viewCookieDetails = function(cookieName) {
    console.log('View cookie details:', cookieName);
    // Implementation would show cookie detail modal
};

window.viewRuleDetails = function(ruleId) {
    console.log('View rule details:', ruleId);
    // Implementation would show rule detail modal
};