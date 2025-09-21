/**
 * Tag Sentinel UI - Page Detail JavaScript
 * Handles page-level analysis view with timeline and artifacts
 */

// Page state
let runId = null;
let pageId = null;
let timelineData = null;
let zoomLevel = 1;

// Initialize page detail
document.addEventListener('DOMContentLoaded', function() {
    initializePageDetail();
});

/**
 * Initialize page detail page
 */
function initializePageDetail() {
    // Extract IDs from URL
    const pathParts = window.location.pathname.split('/');
    runId = pathParts[pathParts.indexOf('runs') + 1];
    pageId = pathParts[pathParts.indexOf('pages') + 1];

    if (!runId || !pageId) {
        showError('Invalid run or page ID');
        return;
    }

    setupEventListeners();
    loadPageData();

    console.log('Page detail initialized for run:', runId, 'page:', pageId);
}

/**
 * Set up event listeners
 */
function setupEventListeners() {
    // Refresh button
    const refreshBtn = document.getElementById('refresh-btn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', function() {
            loadPageData();
            window.TagSentinel.utils.announceToScreenReader('Refreshing page data');
        });
    }

    // Export button
    const exportBtn = document.getElementById('export-page-btn');
    if (exportBtn) {
        exportBtn.addEventListener('click', exportPageData);
    }

    // View artifacts button
    const artifactsBtn = document.getElementById('view-artifacts-btn');
    if (artifactsBtn) {
        artifactsBtn.addEventListener('click', viewArtifacts);
    }

    // Timeline controls
    setupTimelineControls();

    // Data layer controls
    setupDataLayerControls();
}

/**
 * Set up timeline controls
 */
function setupTimelineControls() {
    const zoomInBtn = document.getElementById('zoom-in-btn');
    const zoomOutBtn = document.getElementById('zoom-out-btn');
    const zoomResetBtn = document.getElementById('zoom-reset-btn');

    if (zoomInBtn) {
        zoomInBtn.addEventListener('click', () => adjustZoom(1.5));
    }

    if (zoomOutBtn) {
        zoomOutBtn.addEventListener('click', () => adjustZoom(0.75));
    }

    if (zoomResetBtn) {
        zoomResetBtn.addEventListener('click', () => resetZoom());
    }

    // Timeline filter
    const timelineFilter = document.getElementById('timeline-filter');
    if (timelineFilter) {
        timelineFilter.addEventListener('change', filterTimeline);
    }

    // Timeline search
    const timelineSearch = document.getElementById('timeline-search');
    if (timelineSearch) {
        timelineSearch.addEventListener('input',
            window.TagSentinel.utils.debounce(searchTimeline, 300)
        );
    }
}

/**
 * Set up data layer controls
 */
function setupDataLayerControls() {
    const expandAllBtn = document.getElementById('expand-all-btn');
    const collapseAllBtn = document.getElementById('collapse-all-btn');
    const validateBtn = document.getElementById('validate-schema-btn');

    if (expandAllBtn) {
        expandAllBtn.addEventListener('click', () => toggleDataLayerAll(true));
    }

    if (collapseAllBtn) {
        collapseAllBtn.addEventListener('click', () => toggleDataLayerAll(false));
    }

    if (validateBtn) {
        validateBtn.addEventListener('click', validateDataLayerSchema);
    }
}

/**
 * Load page data
 */
async function loadPageData() {
    try {
        showLoadingStates();

        // Get audit details and extract page-specific data
        const auditData = await window.TagSentinel.api.getAuditDetail(runId);

        // Find the specific page in the audit data
        const pageData = auditData.pages?.find(page => page.id === pageId) || {};

        // Extract page-specific data from audit results and structure for display functions
        const timelineData = { requests: pageData.timeline || [] };
        const detectionsData = { detections: pageData.detections || [] };
        const dataLayerData = pageData.data_layer || {};
        const cookiesData = { cookies: pageData.cookies || [] };
        const consoleData = { messages: pageData.console_logs || [] };
        const artifactsData = { artifacts: pageData.artifacts || [] };

        displayPageInfo(pageData);
        displayTimeline(timelineData);
        displayDetections(detectionsData);
        displayDataLayer(dataLayerData);
        displayCookies(cookiesData);
        displayConsole(consoleData);
        displayArtifacts(artifactsData);

    } catch (error) {
        console.error('Failed to load page data:', error);
        showPageError(error.message);
    }
}

/**
 * Display page info
 */
function displayPageInfo(data) {
    const pageInfo = document.getElementById('page-info');
    if (!pageInfo || !data) return;

    const loadTime = data.load_time || 0;
    const status = data.status || 'unknown';

    pageInfo.innerHTML = `
        <div class="page-url">
            <strong>URL:</strong> ${escapeHtml(data.url || 'Unknown')}
        </div>
        <div class="page-details-grid">
            <div class="page-detail-item">
                <span class="page-detail-label">Status</span>
                <div class="page-detail-value">
                    <span class="badge badge-${getStatusClass(status)}">${status}</span>
                </div>
            </div>
            <div class="page-detail-item">
                <span class="page-detail-label">Load Time</span>
                <div class="page-detail-value">${formatLoadTime(loadTime)}</div>
            </div>
            <div class="page-detail-item">
                <span class="page-detail-label">Tags Detected</span>
                <div class="page-detail-value">${data.tags_count || 0}</div>
            </div>
            <div class="page-detail-item">
                <span class="page-detail-label">Issues Found</span>
                <div class="page-detail-value ${data.issues_count > 0 ? 'text-danger' : 'text-success'}">
                    ${data.issues_count || 0}
                </div>
            </div>
            <div class="page-detail-item">
                <span class="page-detail-label">Network Requests</span>
                <div class="page-detail-value">${data.requests_count || 0}</div>
            </div>
            <div class="page-detail-item">
                <span class="page-detail-label">Console Errors</span>
                <div class="page-detail-value ${data.console_errors > 0 ? 'text-warning' : 'text-success'}">
                    ${data.console_errors || 0}
                </div>
            </div>
        </div>
    `;

    // Update page title
    if (data.url) {
        document.title = `${data.url} - Page Detail - Tag Sentinel`;
    }
}

/**
 * Display timeline
 */
function displayTimeline(data) {
    const timelineContent = document.getElementById('timeline-content');
    if (!timelineContent || !data.requests) return;

    timelineData = data;

    // Helper function to sanitize timing values
    function sanitizeTime(value) {
        const num = Number(value);
        return Number.isFinite(num) && num >= 0 ? num : null;
    }

    // Sanitize all request times first
    const sanitizedRequests = data.requests.map(request => {
        const sanitizedStartTime = sanitizeTime(request.start_time) || 0;
        const rawSanitizedEndTime = sanitizeTime(request.end_time);

        // Ensure end time is at least as large as start time
        const sanitizedEndTime = rawSanitizedEndTime != null
            ? Math.max(rawSanitizedEndTime, sanitizedStartTime)
            : sanitizedStartTime;

        return {
            ...request,
            sanitizedStartTime,
            sanitizedEndTime
        };
    });

    // Calculate max time using sanitized values
    let maxTime = 0;
    if (sanitizedRequests.length > 0) {
        const times = sanitizedRequests
            .map(r => r.sanitizedEndTime)
            .filter(time => time > 0);
        maxTime = times.length > 0 ? Math.max(...times) : 1000; // Default to 1 second
    } else {
        maxTime = 1000; // Default to 1 second for empty requests
    }

    // Update ruler
    updateTimelineRuler(maxTime);

    // Render timeline items using sanitized values
    timelineContent.innerHTML = sanitizedRequests.map((request, index) => {
        const startTime = request.sanitizedStartTime;
        const endTime = request.sanitizedEndTime;

        const startPercent = (startTime / maxTime) * 100 * zoomLevel;
        const duration = Math.max(endTime - startTime, 0); // Ensure non-negative duration
        const widthPercent = Math.max((duration / maxTime) * 100 * zoomLevel, 1);

        return `
            <div class="timeline-row" data-request-index="${index}">
                <div class="timeline-label" title="${escapeHtml(request.url)}">
                    ${escapeHtml(request.name || request.url)}
                </div>
                <div class="timeline-bar ${getRequestType(request)}"
                     style="left: ${startPercent}%; width: ${widthPercent}%;"
                     onclick="showRequestDetails(${index})"
                     title="${escapeHtml(request.url)} - ${formatDuration(duration)}">
                </div>
                <div class="timeline-info">
                    ${formatDuration(duration)}
                </div>
            </div>
        `;
    }).join('');
}

/**
 * Update timeline ruler
 */
function updateTimelineRuler(maxTime) {
    const ruler = document.getElementById('timeline-ruler');
    if (!ruler) return;

    const markers = 10;
    const markerInterval = maxTime / markers;

    ruler.innerHTML = Array.from({ length: markers + 1 }, (_, i) => {
        const time = i * markerInterval;
        const position = (time / maxTime) * 100 * zoomLevel;

        return `
            <div class="timeline-marker" style="left: ${position}%">
                ${formatTime(time)}
            </div>
        `;
    }).join('');
}

/**
 * Display detections
 */
function displayDetections(data) {
    const tbody = document.getElementById('detections-tbody');
    if (!tbody || !data.detections) return;

    tbody.innerHTML = data.detections.map(detection => `
        <tr>
            <td>
                <span class="vendor-badge vendor-${detection.vendor || 'unknown'}">
                    ${escapeHtml((detection.vendor || 'unknown').toUpperCase())}
                </span>
            </td>
            <td>${escapeHtml(detection.tag_type || 'Unknown')}</td>
            <td>${escapeHtml(detection.measurement_id || '—')}</td>
            <td>
                <span class="confidence-badge confidence-${detection.confidence}">
                    ${escapeHtml(detection.confidence || 'unknown')}
                </span>
            </td>
            <td>${detection.events_count || 0}</td>
            <td>
                <button class="btn btn-secondary btn-sm" onclick="viewDetectionDetails('${detection.id}')">
                    Details
                </button>
            </td>
        </tr>
    `).join('');
}

/**
 * Display data layer
 */
function displayDataLayer(data) {
    const summary = document.getElementById('datalayer-summary');
    const viewer = document.getElementById('datalayer-viewer');

    if (summary && data.summary) {
        summary.innerHTML = `
            <div class="datalayer-stat">
                <span class="datalayer-stat-label">Variables</span>
                <div class="datalayer-stat-value">${data.summary.variables_count || 0}</div>
            </div>
            <div class="datalayer-stat">
                <span class="datalayer-stat-label">Events</span>
                <div class="datalayer-stat-value">${data.summary.events_count || 0}</div>
            </div>
            <div class="datalayer-stat">
                <span class="datalayer-stat-label">Schema Valid</span>
                <div class="datalayer-stat-value ${data.summary.schema_valid ? 'text-success' : 'text-danger'}">
                    ${data.summary.schema_valid ? '✓' : '✗'}
                </div>
            </div>
            <div class="datalayer-stat">
                <span class="datalayer-stat-label">Size</span>
                <div class="datalayer-stat-value">${formatBytes(data.summary.size_bytes || 0)}</div>
            </div>
        `;
    }

    if (viewer && data.snapshot) {
        viewer.innerHTML = `
            <div class="json-viewer">
                ${formatJsonForDisplay(data.snapshot)}
            </div>
        `;
    }
}

/**
 * Display cookies
 */
function displayCookies(data) {
    const tbody = document.getElementById('cookies-tbody');
    if (!tbody || !data.cookies) return;

    tbody.innerHTML = data.cookies.map(cookie => `
        <tr>
            <td>${escapeHtml(cookie.name)}</td>
            <td>${escapeHtml(cookie.domain)}</td>
            <td>
                <span class="category-badge category-${cookie.category}">
                    ${escapeHtml(cookie.category)}
                </span>
            </td>
            <td>${formatDuration(cookie.max_age || 0)}</td>
            <td>${formatBytes(cookie.size || 0)}</td>
            <td>
                <span class="privacy-impact impact-${cookie.privacy_impact}">
                    ${escapeHtml(cookie.privacy_impact || 'low')}
                </span>
            </td>
        </tr>
    `).join('');
}

/**
 * Display console messages
 */
function displayConsole(data) {
    const consoleContent = document.getElementById('console-content');
    if (!consoleContent || !data.messages) return;

    consoleContent.innerHTML = data.messages.map(message => `
        <div class="console-message ${message.level}">
            <span class="console-timestamp">${formatTime(message.timestamp)}</span>
            <span class="console-source">${escapeHtml(message.source || '')}</span>
            <div class="console-text">${escapeHtml(message.text)}</div>
        </div>
    `).join('');
}

/**
 * Display artifacts
 */
function displayArtifacts(data) {
    const artifactsGrid = document.getElementById('artifacts-grid');
    if (!artifactsGrid || !data.artifacts) return;

    artifactsGrid.innerHTML = data.artifacts.map(artifact => `
        <div class="artifact-card">
            <div class="artifact-icon">${getArtifactIcon(artifact.type)}</div>
            <div class="artifact-name">${escapeHtml(artifact.name)}</div>
            <div class="artifact-size">${formatBytes(artifact.size || 0)}</div>
            <div class="artifact-actions">
                <a href="${artifact.download_url}" class="btn btn-secondary btn-sm" download>
                    Download
                </a>
                ${artifact.viewable ? `
                    <button class="btn btn-secondary btn-sm" onclick="viewArtifact('${artifact.id}')">
                        View
                    </button>
                ` : ''}
            </div>
        </div>
    `).join('');
}

/**
 * Helper functions
 */
function getStatusClass(status) {
    const statusMap = {
        'success': 'success', 'completed': 'success',
        'completed_with_issues': 'warning',
        'warning': 'warning',
        'error': 'danger', 'failed': 'danger',
        'unknown': 'secondary'
    };
    return statusMap[status] || 'secondary';
}

function getRequestType(request) {
    // Guard against missing URL
    const url = request.url || '';

    if (url.includes('google-analytics') || url.includes('gtag')) return 'analytics';
    if (url.includes('facebook') || url.includes('doubleclick')) return 'marketing';
    if (url.match(/\.(jpg|jpeg|png|gif|webp)$/i)) return 'images';
    if (url.match(/\.(js|css)$/i)) return 'scripts';
    if (request.type === 'xhr' || request.type === 'fetch') return 'xhr';
    return 'other';
}

function getArtifactIcon(type) {
    const icons = {
        'screenshot': '📷',
        'har': '📄',
        'trace': '🔍',
        'video': '🎥',
        'log': '📝'
    };
    return icons[type] || '📎';
}

function formatLoadTime(milliseconds) {
    if (!milliseconds) return '—';
    if (milliseconds < 1000) return `${milliseconds}ms`;
    return `${(milliseconds / 1000).toFixed(1)}s`;
}

function formatDuration(milliseconds) {
    if (!milliseconds) return '0ms';
    if (milliseconds < 1000) return `${Math.round(milliseconds)}ms`;
    return `${(milliseconds / 1000).toFixed(1)}s`;
}

function formatTime(milliseconds) {
    if (!milliseconds) return '0ms';
    if (milliseconds < 1000) return `${Math.round(milliseconds)}ms`;
    if (milliseconds < 60000) return `${(milliseconds / 1000).toFixed(1)}s`;
    return `${Math.floor(milliseconds / 60000)}:${((milliseconds % 60000) / 1000).toFixed(0).padStart(2, '0')}`;
}

function formatBytes(bytes) {
    if (!bytes) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

function formatJsonForDisplay(obj, indent = 0) {
    if (obj === null) return '<span class="json-null">null</span>';
    if (typeof obj === 'string') return `<span class="json-string">"${escapeHtml(obj)}"</span>`;
    if (typeof obj === 'number') return `<span class="json-number">${obj}</span>`;
    if (typeof obj === 'boolean') return `<span class="json-boolean">${obj}</span>`;

    const spaces = '  '.repeat(indent);

    if (Array.isArray(obj)) {
        if (obj.length === 0) return '[]';
        return `[
${obj.map(item => `${spaces}  ${formatJsonForDisplay(item, indent + 1)}`).join(',\n')}
${spaces}]`;
    }

    if (typeof obj === 'object') {
        const keys = Object.keys(obj);
        if (keys.length === 0) return '{}';
        return `{
${keys.map(key => `${spaces}  <span class="json-key">"${escapeHtml(key)}"</span>: ${formatJsonForDisplay(obj[key], indent + 1)}`).join(',\n')}
${spaces}}`;
    }

    return String(obj);
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showLoadingStates() {
    const sections = ['page-info', 'timeline-content', 'detections-tbody', 'datalayer-viewer', 'cookies-tbody', 'console-content', 'artifacts-grid'];

    sections.forEach(sectionId => {
        const element = document.getElementById(sectionId);
        if (element) {
            element.innerHTML = `
                <div class="loading-container">
                    <div class="spinner" aria-hidden="true"></div>
                    <span>Loading...</span>
                </div>
            `;
        }
    });
}

function showPageError(message) {
    const pageInfo = document.getElementById('page-info');
    if (pageInfo) {
        pageInfo.innerHTML = `
            <div class="error-message">
                <p>Unable to load page data: ${escapeHtml(message)}</p>
                <button onclick="loadPageData()" class="btn btn-primary">Retry</button>
            </div>
        `;
    }
}

// Event handlers
function adjustZoom(factor) {
    zoomLevel *= factor;
    zoomLevel = Math.max(0.25, Math.min(zoomLevel, 4));

    if (timelineData) {
        displayTimeline(timelineData);
    }
}

function resetZoom() {
    zoomLevel = 1;
    if (timelineData) {
        displayTimeline(timelineData);
    }
}

function filterTimeline() {
    // Implementation for timeline filtering
    console.log('Filter timeline');
}

function searchTimeline() {
    // Implementation for timeline search
    console.log('Search timeline');
}

function toggleDataLayerAll(expand) {
    // Implementation for expand/collapse all
    console.log('Toggle data layer:', expand);
}

function validateDataLayerSchema() {
    // Implementation for schema validation
    console.log('Validate schema');
}

function exportPageData() {
    // Implementation for page data export
    console.log('Export page data');
}

function viewArtifacts() {
    // Implementation for artifacts viewer
    console.log('View artifacts');
}

function showRequestDetails(index) {
    // Implementation for request details modal
    console.log('Show request details:', index);
}

function viewDetectionDetails(id) {
    // Implementation for detection details
    console.log('View detection details:', id);
}

function viewArtifact(id) {
    // Implementation for artifact viewer
    console.log('View artifact:', id);
}

function showError(message) {
    console.error(message);
    window.TagSentinel.utils.announceToScreenReader(`Error: ${message}`);
}