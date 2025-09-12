"""Global Privacy Control (GPC) simulation and analysis.

This module implements GPC signal simulation including header injection,
JavaScript API simulation, and response analysis to determine site compliance
with Global Privacy Control signals.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from playwright.async_api import BrowserContext, Page, Route, Request

from .models import CookieRecord
from .config import GPCConfig, PrivacyConfiguration

logger = logging.getLogger(__name__)


class GPCResponseAnalysis:
    """Analysis of how a site responds to GPC signals."""
    
    def __init__(self):
        self.gpc_aware = False
        self.gpc_javascript_api_present = False
        self.gpc_respect_indicators = []
        self.privacy_policy_links = []
        self.opt_out_mechanisms = []
        self.analysis_time = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis to dictionary."""
        return {
            'gpc_aware': self.gpc_aware,
            'javascript_api_present': self.gpc_javascript_api_present,
            'respect_indicators': self.gpc_respect_indicators,
            'privacy_policy_links': self.privacy_policy_links,
            'opt_out_mechanisms': self.opt_out_mechanisms,
            'analysis_time': self.analysis_time.isoformat()
        }


class GPCSimulator:
    """Global Privacy Control signal simulator and analyzer.
    
    Implements GPC header injection, JavaScript API simulation, and analyzes
    site responses to determine GPC compliance and effectiveness.
    """
    
    def __init__(self, config: Optional[GPCConfig] = None):
        """Initialize GPC simulator.
        
        Args:
            config: GPC configuration settings
        """
        self.config = config or GPCConfig()
        self.is_enabled = False
        self.injected_contexts = set()
        
        # JavaScript to inject for GPC API simulation
        self.gpc_javascript = """
        // Global Privacy Control API simulation
        if (typeof navigator.globalPrivacyControl === 'undefined') {
            Object.defineProperty(navigator, 'globalPrivacyControl', {
                value: true,
                writable: false,
                enumerable: true,
                configurable: false
            });
        }
        
        // Add event listeners for GPC detection
        window.__gpc_events = window.__gpc_events || [];
        
        // Monitor for GPC checks
        const originalGetItem = localStorage.getItem;
        const originalSetItem = localStorage.setItem;
        
        localStorage.getItem = function(key) {
            if (key.toLowerCase().includes('gpc') || key.toLowerCase().includes('privacy')) {
                window.__gpc_events.push({
                    type: 'localStorage_read',
                    key: key,
                    timestamp: Date.now()
                });
            }
            return originalGetItem.apply(this, arguments);
        };
        
        localStorage.setItem = function(key, value) {
            if (key.toLowerCase().includes('gpc') || key.toLowerCase().includes('privacy')) {
                window.__gpc_events.push({
                    type: 'localStorage_write',
                    key: key,
                    value: value,
                    timestamp: Date.now()
                });
            }
            return originalSetItem.apply(this, arguments);
        };
        """
    
    async def enable_gpc_for_context(self, context: BrowserContext) -> None:
        """Enable GPC simulation for a browser context.
        
        Args:
            context: Playwright browser context
        """
        if not self.config.enabled:
            logger.debug("GPC simulation is disabled in configuration")
            return
        
        context_id = id(context)
        if context_id in self.injected_contexts:
            logger.debug("GPC already enabled for this context")
            return
        
        # Set up request interception to inject GPC header
        async def inject_gpc_header(route: Route, request: Request) -> None:
            """Inject GPC header into requests."""
            headers = dict(request.headers)
            
            # Parse header from config (e.g., "Sec-GPC: 1")
            if ':' in self.config.header:
                header_name, header_value = self.config.header.split(':', 1)
                headers[header_name.strip()] = header_value.strip()
            else:
                headers['Sec-GPC'] = '1'  # Default
            
            await route.continue_(headers=headers)
        
        # Enable request interception
        await context.route("**/*", inject_gpc_header)
        
        # Inject JavaScript API if enabled
        if self.config.simulate_javascript_api:
            await context.add_init_script(self.gpc_javascript)
        
        self.injected_contexts.add(context_id)
        self.is_enabled = True
        
        logger.info(f"GPC simulation enabled for context with header: {self.config.header}")
    
    async def disable_gpc_for_context(self, context: BrowserContext) -> None:
        """Disable GPC simulation for a browser context.
        
        Args:
            context: Playwright browser context
        """
        context_id = id(context)
        if context_id not in self.injected_contexts:
            return
        
        # Remove request interception
        await context.unroute("**/*")
        
        self.injected_contexts.discard(context_id)
        
        logger.info("GPC simulation disabled for context")
    
    async def analyze_gpc_response(self, page: Page, page_url: str) -> GPCResponseAnalysis:
        """Analyze how the page responds to GPC signals.
        
        Args:
            page: Playwright page object
            page_url: URL of the page being analyzed
            
        Returns:
            Analysis of GPC response behavior
        """
        analysis = GPCResponseAnalysis()
        
        try:
            # Check for GPC JavaScript API
            gpc_api_result = await page.evaluate("""
                () => {
                    return {
                        globalPrivacyControl: typeof navigator.globalPrivacyControl !== 'undefined',
                        gpcValue: navigator.globalPrivacyControl,
                        gpcEvents: window.__gpc_events || []
                    };
                }
            """)
            
            analysis.gpc_javascript_api_present = gpc_api_result.get('globalPrivacyControl', False)
            
            # Check for GPC awareness indicators in the page
            gpc_indicators = await self._detect_gpc_awareness(page)
            analysis.gpc_respect_indicators = gpc_indicators
            
            # Look for privacy policy links
            privacy_links = await self._find_privacy_policy_links(page)
            analysis.privacy_policy_links = privacy_links
            
            # Check for opt-out mechanisms
            opt_out_mechanisms = await self._detect_opt_out_mechanisms(page)
            analysis.opt_out_mechanisms = opt_out_mechanisms
            
            # Determine overall GPC awareness
            analysis.gpc_aware = (
                len(gpc_indicators) > 0 or 
                gpc_api_result.get('gpcEvents', []) or
                'gpc' in page_url.lower()  # Simple heuristic
            )
            
            logger.info(f"GPC analysis complete for {page_url}: aware={analysis.gpc_aware}")
            
        except Exception as e:
            logger.error(f"Error analyzing GPC response: {e}")
        
        return analysis
    
    async def _detect_gpc_awareness(self, page: Page) -> List[str]:
        """Detect indicators that the site is aware of GPC.
        
        Args:
            page: Playwright page object
            
        Returns:
            List of GPC awareness indicators found
        """
        indicators = []
        
        try:
            # Check for GPC-related text in the page
            gpc_text_result = await page.evaluate("""
                () => {
                    const text = document.body.innerText.toLowerCase();
                    const indicators = [];
                    
                    if (text.includes('global privacy control')) {
                        indicators.push('text_global_privacy_control');
                    }
                    if (text.includes('gpc')) {
                        indicators.push('text_gpc_abbreviation');
                    }
                    if (text.includes('do not sell')) {
                        indicators.push('text_do_not_sell');
                    }
                    if (text.includes('opt out')) {
                        indicators.push('text_opt_out');
                    }
                    if (text.includes('privacy signal')) {
                        indicators.push('text_privacy_signal');
                    }
                    
                    return indicators;
                }
            """)
            
            indicators.extend(gpc_text_result)
            
            # Check for GPC-related elements
            gpc_elements = await page.evaluate("""
                () => {
                    const indicators = [];
                    
                    // Check for elements with GPC-related IDs or classes
                    if (document.querySelector('[id*="gpc"], [class*="gpc"]')) {
                        indicators.push('element_gpc_selector');
                    }
                    
                    // Check for privacy control elements
                    if (document.querySelector('[id*="privacy"], [class*="privacy"]')) {
                        indicators.push('element_privacy_selector');
                    }
                    
                    // Check for opt-out buttons
                    const optOutButtons = document.querySelectorAll('button, a, input');
                    for (let element of optOutButtons) {
                        const text = element.innerText.toLowerCase();
                        if (text.includes('opt out') || text.includes('do not sell')) {
                            indicators.push('element_opt_out_button');
                            break;
                        }
                    }
                    
                    return indicators;
                }
            """)
            
            indicators.extend(gpc_elements)
            
            # Check network requests for GPC-related activity
            network_indicators = await self._check_network_for_gpc_activity(page)
            indicators.extend(network_indicators)
            
        except Exception as e:
            logger.warning(f"Error detecting GPC awareness: {e}")
        
        return list(set(indicators))  # Remove duplicates
    
    async def _check_network_for_gpc_activity(self, page: Page) -> List[str]:
        """Check network activity for GPC-related requests.
        
        Args:
            page: Playwright page object
            
        Returns:
            List of network-based GPC indicators
        """
        indicators = []
        
        try:
            # Get network logs if available (simplified check)
            # In a full implementation, would monitor network requests during page load
            
            # Check for GPC-related URLs in links
            gpc_links = await page.evaluate("""
                () => {
                    const indicators = [];
                    const links = document.querySelectorAll('a[href]');
                    
                    for (let link of links) {
                        const href = link.href.toLowerCase();
                        if (href.includes('gpc') || href.includes('privacy') || 
                            href.includes('opt-out') || href.includes('do-not-sell')) {
                            indicators.push('link_gpc_related_url');
                            break;
                        }
                    }
                    
                    return indicators;
                }
            """)
            
            indicators.extend(gpc_links)
            
        except Exception as e:
            logger.warning(f"Error checking network for GPC activity: {e}")
        
        return indicators
    
    async def _find_privacy_policy_links(self, page: Page) -> List[Dict[str, str]]:
        """Find privacy policy links on the page.
        
        Args:
            page: Playwright page object
            
        Returns:
            List of privacy policy links found
        """
        try:
            privacy_links = await page.evaluate("""
                () => {
                    const links = [];
                    const anchors = document.querySelectorAll('a[href]');
                    
                    for (let anchor of anchors) {
                        const text = anchor.innerText.toLowerCase();
                        const href = anchor.href.toLowerCase();
                        
                        if (text.includes('privacy') || text.includes('policy') ||
                            href.includes('privacy') || href.includes('policy')) {
                            links.push({
                                text: anchor.innerText,
                                href: anchor.href,
                                type: 'privacy_policy'
                            });
                        }
                    }
                    
                    return links;
                }
            """)
            
            return privacy_links
            
        except Exception as e:
            logger.warning(f"Error finding privacy policy links: {e}")
            return []
    
    async def _detect_opt_out_mechanisms(self, page: Page) -> List[Dict[str, str]]:
        """Detect opt-out mechanisms on the page.
        
        Args:
            page: Playwright page object
            
        Returns:
            List of opt-out mechanisms found
        """
        try:
            opt_out_mechanisms = await page.evaluate("""
                () => {
                    const mechanisms = [];
                    
                    // Check for opt-out buttons
                    const buttons = document.querySelectorAll('button, input[type="button"], input[type="submit"]');
                    for (let button of buttons) {
                        const text = button.innerText.toLowerCase() || button.value.toLowerCase();
                        if (text.includes('opt out') || text.includes('do not sell') ||
                            text.includes('reject') || text.includes('decline')) {
                            mechanisms.push({
                                type: 'button',
                                text: button.innerText || button.value,
                                selector: button.tagName.toLowerCase() + 
                                         (button.id ? '#' + button.id : '') +
                                         (button.className ? '.' + button.className.split(' ').join('.') : '')
                            });
                        }
                    }
                    
                    // Check for opt-out links
                    const links = document.querySelectorAll('a[href]');
                    for (let link of links) {
                        const text = link.innerText.toLowerCase();
                        const href = link.href.toLowerCase();
                        if (text.includes('opt out') || text.includes('do not sell') ||
                            href.includes('opt-out') || href.includes('do-not-sell')) {
                            mechanisms.push({
                                type: 'link',
                                text: link.innerText,
                                href: link.href
                            });
                        }
                    }
                    
                    // Check for forms with opt-out functionality
                    const forms = document.querySelectorAll('form');
                    for (let form of forms) {
                        const formText = form.innerText.toLowerCase();
                        if (formText.includes('opt out') || formText.includes('do not sell')) {
                            mechanisms.push({
                                type: 'form',
                                action: form.action,
                                method: form.method
                            });
                        }
                    }
                    
                    return mechanisms;
                }
            """)
            
            return opt_out_mechanisms
            
        except Exception as e:
            logger.warning(f"Error detecting opt-out mechanisms: {e}")
            return []
    
    def analyze_cookie_differences(
        self, 
        baseline_cookies: List[CookieRecord], 
        gpc_cookies: List[CookieRecord]
    ) -> Dict[str, Any]:
        """Analyze differences between baseline and GPC cookie sets.
        
        Args:
            baseline_cookies: Cookies collected without GPC
            gpc_cookies: Cookies collected with GPC enabled
            
        Returns:
            Analysis of GPC effectiveness based on cookie differences
        """
        # Create lookup maps
        baseline_map = {
            f"{c.name}@{c.domain}{c.path}": c for c in baseline_cookies
        }
        gpc_map = {
            f"{c.name}@{c.domain}{c.path}": c for c in gpc_cookies
        }
        
        # Find differences
        removed_cookies = []
        added_cookies = []
        modified_cookies = []
        
        for key, cookie in baseline_map.items():
            if key not in gpc_map:
                removed_cookies.append(cookie)
        
        for key, cookie in gpc_map.items():
            if key not in baseline_map:
                added_cookies.append(cookie)
            else:
                # Check for modifications
                baseline_cookie = baseline_map[key]
                if (cookie.value != baseline_cookie.value or
                    cookie.expires != baseline_cookie.expires):
                    modified_cookies.append({
                        'cookie': cookie,
                        'baseline': baseline_cookie
                    })
        
        # Analyze effectiveness
        total_baseline = len(baseline_cookies)
        total_removed = len(removed_cookies)
        reduction_percentage = (total_removed / total_baseline * 100) if total_baseline > 0 else 0
        
        # Categorize removed cookies
        removed_analytics = [c for c in removed_cookies 
                           if c.metadata and c.metadata.get('classification', {}).get('category') == 'analytics']
        removed_marketing = [c for c in removed_cookies 
                           if c.metadata and c.metadata.get('classification', {}).get('category') == 'marketing']
        
        # Determine effectiveness level
        if reduction_percentage >= 50:
            effectiveness = "high"
        elif reduction_percentage >= 20:
            effectiveness = "medium"
        elif reduction_percentage >= 5:
            effectiveness = "low"
        else:
            effectiveness = "none"
        
        return {
            'effectiveness': effectiveness,
            'reduction_percentage': round(reduction_percentage, 2),
            'baseline_cookies': total_baseline,
            'gpc_cookies': len(gpc_cookies),
            'removed_cookies': total_removed,
            'added_cookies': len(added_cookies),
            'modified_cookies': len(modified_cookies),
            'cookie_analysis': {
                'removed_analytics': len(removed_analytics),
                'removed_marketing': len(removed_marketing),
                'removed_third_party': len([c for c in removed_cookies if not c.is_first_party]),
                'removed_non_essential': len([c for c in removed_cookies if c.essential is False]),
            },
            'detailed_changes': {
                'removed': [{'name': c.name, 'domain': c.domain, 'essential': c.essential} 
                           for c in removed_cookies],
                'added': [{'name': c.name, 'domain': c.domain, 'essential': c.essential} 
                         for c in added_cookies],
            }
        }
    
    def generate_gpc_report(
        self, 
        page_url: str,
        response_analysis: GPCResponseAnalysis,
        cookie_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive GPC analysis report.
        
        Args:
            page_url: URL analyzed
            response_analysis: GPC response analysis
            cookie_analysis: Cookie difference analysis
            
        Returns:
            Comprehensive GPC effectiveness report
        """
        # Overall GPC score based on various factors
        gpc_score = 0
        max_score = 100
        
        # Site awareness (30 points)
        if response_analysis.gpc_aware:
            gpc_score += 30
        
        # JavaScript API support (20 points)
        if response_analysis.gpc_javascript_api_present:
            gpc_score += 20
        
        # Cookie reduction effectiveness (50 points)
        effectiveness_scores = {
            'high': 50,
            'medium': 35,
            'low': 15,
            'none': 0
        }
        gpc_score += effectiveness_scores.get(cookie_analysis['effectiveness'], 0)
        
        return {
            'page_url': page_url,
            'analysis_time': datetime.utcnow().isoformat(),
            'gpc_score': min(gpc_score, max_score),
            'site_gpc_awareness': {
                'is_aware': response_analysis.gpc_aware,
                'javascript_api_present': response_analysis.gpc_javascript_api_present,
                'awareness_indicators': response_analysis.gpc_respect_indicators,
                'privacy_policy_links': len(response_analysis.privacy_policy_links),
                'opt_out_mechanisms': len(response_analysis.opt_out_mechanisms),
            },
            'cookie_effectiveness': cookie_analysis,
            'compliance_assessment': {
                'respects_gpc': cookie_analysis['effectiveness'] in ['high', 'medium'],
                'recommendation': self._generate_gpc_recommendation(response_analysis, cookie_analysis),
                'privacy_friendly': gpc_score >= 70,
            },
            'detailed_analysis': {
                'response_analysis': response_analysis.to_dict(),
                'enabled_config': {
                    'header_injected': self.config.header,
                    'javascript_api_simulated': self.config.simulate_javascript_api,
                }
            }
        }
    
    def _generate_gpc_recommendation(
        self, 
        response_analysis: GPCResponseAnalysis, 
        cookie_analysis: Dict[str, Any]
    ) -> str:
        """Generate recommendation based on GPC analysis.
        
        Args:
            response_analysis: GPC response analysis
            cookie_analysis: Cookie difference analysis
            
        Returns:
            Recommendation string
        """
        effectiveness = cookie_analysis['effectiveness']
        
        if effectiveness == 'high':
            return "Site shows good GPC compliance with significant cookie reduction"
        elif effectiveness == 'medium':
            return "Site shows partial GPC compliance but could improve cookie reduction"
        elif effectiveness == 'low':
            return "Site shows minimal GPC compliance - consider improving privacy controls"
        else:
            if response_analysis.gpc_aware:
                return "Site is GPC-aware but not effectively reducing cookies - review implementation"
            else:
                return "Site does not appear to implement GPC - consider adding Global Privacy Control support"