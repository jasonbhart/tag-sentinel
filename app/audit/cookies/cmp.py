"""Consent Management Platform (CMP) detection and automation.

This module provides comprehensive CMP detection, analysis, and automated
consent interaction capabilities for testing different consent states
in privacy scenarios.
"""

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from urllib.parse import urlparse

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError

from .models import ConsentState
from .config import CMPConfig, PrivacyConfiguration

logger = logging.getLogger(__name__)


class CMPPlatform(str, Enum):
    """Known CMP platforms."""
    ONETRUST = "onetrust"
    COOKIEBOT = "cookiebot"
    TRUSTARC = "trustarc"
    QUANTCAST = "quantcast"
    COOKIEPRO = "cookiepro"
    CONSENT_MANAGER = "consent_manager"
    CUSTOM = "custom"
    UNKNOWN = "unknown"


class CMPInteractionResult:
    """Result of CMP interaction attempt."""
    
    def __init__(self):
        self.success = False
        self.consent_state = ConsentState.UNKNOWN
        self.platform = CMPPlatform.UNKNOWN
        self.interaction_steps = []
        self.errors = []
        self.screenshots = []
        self.final_state_verified = False
        self.interaction_time = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'success': self.success,
            'consent_state': self.consent_state.value,
            'platform': self.platform.value,
            'interaction_steps': self.interaction_steps,
            'errors': self.errors,
            'screenshots': self.screenshots,
            'final_state_verified': self.final_state_verified,
            'interaction_time': self.interaction_time.isoformat()
        }


class CMPDetector:
    """Detects and analyzes Consent Management Platform presence and configuration."""
    
    def __init__(self):
        self.detection_selectors = {
            CMPPlatform.ONETRUST: [
                '#onetrust-banner-sdk',
                '#onetrust-consent-sdk',
                '.ot-sdk-container',
                '[data-module-name="onetrust"]'
            ],
            CMPPlatform.COOKIEBOT: [
                '#CybotCookiebotDialog',
                '#CybotCookiebotDialogBody',
                '.CybotCookiebotDialog',
                '[data-cookieconsent]'
            ],
            CMPPlatform.TRUSTARC: [
                '#truste-consent-track',
                '.trustarc-banner',
                '#consent-pref-link'
            ],
            CMPPlatform.QUANTCAST: [
                '.qc-cmp-ui',
                '#qc-cmp-ui',
                '.quantcast-choice'
            ],
            CMPPlatform.CONSENT_MANAGER: [
                '.consentmanager-modal',
                '#cmpbox',
                '.cmpbox'
            ]
        }
        
        # JavaScript indicators for different platforms
        self.js_indicators = {
            CMPPlatform.ONETRUST: [
                'window.OneTrust',
                'window.OnetrustActiveGroups',
                'window.Optanon'
            ],
            CMPPlatform.COOKIEBOT: [
                'window.Cookiebot',
                'window.CookieControl'
            ],
            CMPPlatform.TRUSTARC: [
                'window.truste',
                'window.TrustArc'
            ]
        }
    
    async def detect_cmp(self, page: Page) -> Tuple[CMPPlatform, Dict[str, Any]]:
        """Detect CMP platform and analyze its configuration.
        
        Args:
            page: Playwright page object
            
        Returns:
            Tuple of (detected_platform, analysis_details)
        """
        analysis = {
            'detected_elements': [],
            'javascript_objects': [],
            'modal_present': False,
            'configuration': {}
        }
        
        detected_platform = CMPPlatform.UNKNOWN
        
        try:
            # Check for DOM elements
            for platform, selectors in self.detection_selectors.items():
                for selector in selectors:
                    try:
                        element = await page.query_selector(selector)
                        if element:
                            detected_platform = platform
                            analysis['detected_elements'].append(selector)
                            
                            # Check if modal is visible
                            is_visible = await element.is_visible()
                            if is_visible:
                                analysis['modal_present'] = True
                            
                            break
                    except Exception as e:
                        logger.debug(f"Error checking selector {selector}: {e}")
                
                if detected_platform != CMPPlatform.UNKNOWN:
                    break
            
            # Check for JavaScript objects
            if detected_platform != CMPPlatform.UNKNOWN:
                js_objects = self.js_indicators.get(detected_platform, [])
                for js_obj in js_objects:
                    try:
                        exists = await page.evaluate(f"typeof {js_obj} !== 'undefined'")
                        if exists:
                            analysis['javascript_objects'].append(js_obj)
                    except Exception as e:
                        logger.debug(f"Error checking JS object {js_obj}: {e}")
            
            # Platform-specific analysis
            if detected_platform == CMPPlatform.ONETRUST:
                analysis['configuration'] = await self._analyze_onetrust(page)
            elif detected_platform == CMPPlatform.COOKIEBOT:
                analysis['configuration'] = await self._analyze_cookiebot(page)
            
            # Generic CMP detection if no specific platform found
            if detected_platform == CMPPlatform.UNKNOWN:
                detected_platform, generic_analysis = await self._detect_generic_cmp(page)
                analysis.update(generic_analysis)
            
            logger.info(f"CMP detection result: {detected_platform.value}")
            
        except Exception as e:
            logger.error(f"Error during CMP detection: {e}")
        
        return detected_platform, analysis
    
    async def _analyze_onetrust(self, page: Page) -> Dict[str, Any]:
        """Analyze OneTrust CMP configuration.
        
        Args:
            page: Playwright page object
            
        Returns:
            OneTrust configuration details
        """
        config = {}
        
        try:
            config = await page.evaluate("""
                () => {
                    const analysis = {};
                    
                    if (typeof window.OneTrust !== 'undefined') {
                        analysis.version = window.OneTrust.version || 'unknown';
                        
                        // Check for consent model
                        const consentModel = document.querySelector('[data-consent-model]');
                        if (consentModel) {
                            analysis.consentModel = consentModel.getAttribute('data-consent-model');
                        }
                        
                        // Check for available groups
                        if (window.OnetrustActiveGroups) {
                            analysis.activeGroups = window.OnetrustActiveGroups;
                        }
                    }
                    
                    // Check for banner elements
                    const banner = document.querySelector('#onetrust-banner-sdk');
                    if (banner) {
                        analysis.bannerPresent = true;
                        analysis.bannerVisible = !banner.hidden && banner.style.display !== 'none';
                    }
                    
                    // Check for preference center
                    const prefCenter = document.querySelector('#onetrust-pc-sdk');
                    if (prefCenter) {
                        analysis.preferenceCenter = true;
                    }
                    
                    return analysis;
                }
            """)
        except Exception as e:
            logger.warning(f"Error analyzing OneTrust: {e}")
        
        return config
    
    async def _analyze_cookiebot(self, page: Page) -> Dict[str, Any]:
        """Analyze Cookiebot CMP configuration.
        
        Args:
            page: Playwright page object
            
        Returns:
            Cookiebot configuration details
        """
        config = {}
        
        try:
            config = await page.evaluate("""
                () => {
                    const analysis = {};
                    
                    if (typeof window.Cookiebot !== 'undefined') {
                        analysis.consentLevel = window.Cookiebot.consent || {};
                        analysis.configured = window.Cookiebot.configured || false;
                    }
                    
                    // Check for dialog
                    const dialog = document.querySelector('#CybotCookiebotDialog');
                    if (dialog) {
                        analysis.dialogPresent = true;
                        analysis.dialogVisible = dialog.style.display !== 'none';
                    }
                    
                    return analysis;
                }
            """)
        except Exception as e:
            logger.warning(f"Error analyzing Cookiebot: {e}")
        
        return config
    
    async def _detect_generic_cmp(self, page: Page) -> Tuple[CMPPlatform, Dict[str, Any]]:
        """Detect generic CMP patterns.
        
        Args:
            page: Playwright page object
            
        Returns:
            Tuple of (platform, analysis)
        """
        analysis = {}
        
        try:
            # Look for common cookie consent patterns
            consent_patterns = await page.evaluate("""
                () => {
                    const patterns = [];
                    const text = document.body.innerText.toLowerCase();
                    
                    if (text.includes('cookie') && 
                        (text.includes('consent') || text.includes('accept') || text.includes('agree'))) {
                        patterns.push('cookie_consent_text');
                    }
                    
                    // Look for common button patterns
                    const buttons = document.querySelectorAll('button, a, input[type="button"]');
                    for (let button of buttons) {
                        const buttonText = button.innerText.toLowerCase();
                        if (buttonText.includes('accept cookies') || 
                            buttonText.includes('accept all') ||
                            buttonText.includes('reject cookies') ||
                            buttonText.includes('reject all')) {
                            patterns.push('consent_button');
                            break;
                        }
                    }
                    
                    // Look for privacy policy links
                    const links = document.querySelectorAll('a[href*="privacy"], a[href*="cookie"]');
                    if (links.length > 0) {
                        patterns.push('privacy_policy_link');
                    }
                    
                    return patterns;
                }
            """)
            
            analysis['consent_patterns'] = consent_patterns
            
            if consent_patterns:
                return CMPPlatform.CUSTOM, analysis
            
        except Exception as e:
            logger.warning(f"Error in generic CMP detection: {e}")
        
        return CMPPlatform.UNKNOWN, analysis


class ConsentAutomator:
    """Automated consent interaction system for CMP testing."""
    
    def __init__(self, config: Optional[CMPConfig] = None):
        """Initialize consent automator.
        
        Args:
            config: CMP configuration settings
        """
        self.config = config or CMPConfig()
        self.detector = CMPDetector()
        
        # Platform-specific button selectors
        self.platform_selectors = {
            CMPPlatform.ONETRUST: {
                'accept_all': '#onetrust-accept-btn-handler',
                'reject_all': '#onetrust-reject-all-handler',
                'settings': '#onetrust-pc-btn-handler'
            },
            CMPPlatform.COOKIEBOT: {
                'accept_all': '#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll',
                'reject_all': '#CybotCookiebotDialogBodyLevelButtonLevelOptinDeclineAll',
                'settings': '#CybotCookiebotDialogBodyLevelButtonSettings'
            }
        }
    
    async def execute_consent_scenario(
        self, 
        page: Page, 
        consent_state: ConsentState,
        screenshot_path: Optional[str] = None
    ) -> CMPInteractionResult:
        """Execute a consent scenario (accept all, reject all, etc.).
        
        Args:
            page: Playwright page object
            consent_state: Desired consent state
            screenshot_path: Optional path for screenshots
            
        Returns:
            Result of consent interaction
        """
        result = CMPInteractionResult()
        result.consent_state = consent_state
        
        try:
            # Detect CMP platform
            platform, detection_analysis = await self.detector.detect_cmp(page)
            result.platform = platform
            
            if platform == CMPPlatform.UNKNOWN:
                result.errors.append("No CMP detected on page")
                return result
            
            result.interaction_steps.append(f"Detected CMP platform: {platform.value}")
            
            # Wait for CMP modal to appear
            await self._wait_for_cmp_modal(page, platform)
            result.interaction_steps.append("Waited for CMP modal")
            
            # Take before screenshot
            if screenshot_path:
                before_screenshot = f"{screenshot_path}_before.png"
                await page.screenshot(path=before_screenshot)
                result.screenshots.append(before_screenshot)
            
            # Execute consent interaction
            success = await self._perform_consent_interaction(page, platform, consent_state)
            
            if success:
                result.success = True
                result.interaction_steps.append(f"Successfully executed {consent_state.value} interaction")
                
                # Wait for changes to take effect
                await asyncio.sleep(self.config.wait_after_click_ms / 1000)
                
                # Take after screenshot
                if screenshot_path:
                    after_screenshot = f"{screenshot_path}_after.png"
                    await page.screenshot(path=after_screenshot)
                    result.screenshots.append(after_screenshot)
                
                # Verify consent state if possible
                result.final_state_verified = await self._verify_consent_state(page, platform, consent_state)
                
            else:
                result.errors.append(f"Failed to execute {consent_state.value} interaction")
            
        except Exception as e:
            logger.error(f"Error executing consent scenario: {e}")
            result.errors.append(str(e))
        
        return result
    
    async def _wait_for_cmp_modal(self, page: Page, platform: CMPPlatform) -> bool:
        """Wait for CMP modal to appear.
        
        Args:
            page: Playwright page object
            platform: Detected CMP platform
            
        Returns:
            True if modal appeared, False otherwise
        """
        try:
            # Get platform-specific selectors
            selectors = self.detector.detection_selectors.get(platform, [])
            
            if not selectors:
                # Try generic wait
                await asyncio.sleep(self.config.wait_for_modal_ms / 1000)
                return True
            
            # Wait for any of the selectors to be visible
            for selector in selectors:
                try:
                    await page.wait_for_selector(
                        selector, 
                        state='visible', 
                        timeout=self.config.wait_for_modal_ms
                    )
                    return True
                except PlaywrightTimeoutError:
                    continue
            
            # If no specific selector worked, wait and return true
            await asyncio.sleep(2)  # Give some time for modal to appear
            return True
            
        except Exception as e:
            logger.warning(f"Error waiting for CMP modal: {e}")
            return False
    
    async def _perform_consent_interaction(
        self, 
        page: Page, 
        platform: CMPPlatform, 
        consent_state: ConsentState
    ) -> bool:
        """Perform the actual consent interaction.
        
        Args:
            page: Playwright page object
            platform: CMP platform
            consent_state: Desired consent state
            
        Returns:
            True if interaction succeeded
        """
        try:
            # Get platform-specific selectors
            platform_buttons = self.platform_selectors.get(platform, {})
            config_buttons = self.config.selectors
            
            # Determine target selector based on consent state
            target_selector = None
            
            if consent_state == ConsentState.ACCEPT_ALL:
                target_selector = (
                    platform_buttons.get('accept_all') or 
                    config_buttons.get('accept_all') or
                    config_buttons.get('onetrust_accept_all') or
                    config_buttons.get('cookiebot_accept_all')
                )
            elif consent_state == ConsentState.REJECT_ALL:
                target_selector = (
                    platform_buttons.get('reject_all') or 
                    config_buttons.get('reject_all') or
                    config_buttons.get('onetrust_reject_all') or
                    config_buttons.get('cookiebot_reject_all')
                )
            
            if not target_selector:
                logger.warning(f"No selector found for {consent_state.value} on {platform.value}")
                return await self._try_generic_consent_interaction(page, consent_state)
            
            # Try multiple interaction attempts
            for attempt in range(self.config.max_interaction_attempts):
                try:
                    # Check if element exists and is visible
                    element = await page.query_selector(target_selector)
                    if not element:
                        logger.warning(f"Element not found: {target_selector} (attempt {attempt + 1})")
                        await asyncio.sleep(1)
                        continue
                    
                    is_visible = await element.is_visible()
                    if not is_visible:
                        logger.warning(f"Element not visible: {target_selector} (attempt {attempt + 1})")
                        await asyncio.sleep(1)
                        continue
                    
                    # Click the element
                    await element.click()
                    logger.info(f"Clicked {consent_state.value} button: {target_selector}")
                    
                    # Wait for action to complete
                    await asyncio.sleep(self.config.wait_after_click_ms / 1000)
                    
                    return True
                    
                except Exception as e:
                    logger.warning(f"Click attempt {attempt + 1} failed: {e}")
                    if attempt < self.config.max_interaction_attempts - 1:
                        await asyncio.sleep(1)
            
            return False
            
        except Exception as e:
            logger.error(f"Error performing consent interaction: {e}")
            return False
    
    async def _try_generic_consent_interaction(
        self, 
        page: Page, 
        consent_state: ConsentState
    ) -> bool:
        """Try generic consent interaction patterns.
        
        Args:
            page: Playwright page object
            consent_state: Desired consent state
            
        Returns:
            True if interaction succeeded
        """
        try:
            # Generic text-based selectors
            if consent_state == ConsentState.ACCEPT_ALL:
                text_patterns = ['Accept all', 'Accept All', 'Allow all', 'I agree']
            elif consent_state == ConsentState.REJECT_ALL:
                text_patterns = ['Reject all', 'Reject All', 'Decline all', 'No thanks']
            else:
                return False
            
            # Try to find buttons by text
            for pattern in text_patterns:
                try:
                    # Try button elements first
                    button = await page.query_selector(f'button:has-text("{pattern}")')
                    if button and await button.is_visible():
                        await button.click()
                        logger.info(f"Clicked button with text: {pattern}")
                        return True
                    
                    # Try links
                    link = await page.query_selector(f'a:has-text("{pattern}")')
                    if link and await link.is_visible():
                        await link.click()
                        logger.info(f"Clicked link with text: {pattern}")
                        return True
                    
                except Exception as e:
                    logger.debug(f"Error trying pattern '{pattern}': {e}")
            
            return False
            
        except Exception as e:
            logger.error(f"Error in generic consent interaction: {e}")
            return False
    
    async def _verify_consent_state(
        self, 
        page: Page, 
        platform: CMPPlatform, 
        expected_state: ConsentState
    ) -> bool:
        """Verify that consent state was set correctly.
        
        Args:
            page: Playwright page object
            platform: CMP platform
            expected_state: Expected consent state
            
        Returns:
            True if state verified successfully
        """
        try:
            # Platform-specific verification
            if platform == CMPPlatform.ONETRUST:
                return await self._verify_onetrust_state(page, expected_state)
            elif platform == CMPPlatform.COOKIEBOT:
                return await self._verify_cookiebot_state(page, expected_state)
            else:
                # Generic verification - check if modal is gone
                modal_gone = await self._check_modal_dismissed(page, platform)
                return modal_gone
            
        except Exception as e:
            logger.warning(f"Error verifying consent state: {e}")
            return False
    
    async def _verify_onetrust_state(self, page: Page, expected_state: ConsentState) -> bool:
        """Verify OneTrust consent state.
        
        Args:
            page: Playwright page object
            expected_state: Expected consent state
            
        Returns:
            True if state matches expectation
        """
        try:
            consent_data = await page.evaluate("""
                () => {
                    if (typeof window.OneTrust === 'undefined') return null;
                    
                    return {
                        activeGroups: window.OnetrustActiveGroups || '',
                        bannerVisible: document.querySelector('#onetrust-banner-sdk') ? 
                                     !document.querySelector('#onetrust-banner-sdk').hidden : false
                    };
                }
            """)
            
            if not consent_data:
                return False
            
            # Check if banner was dismissed (good sign)
            if consent_data['bannerVisible']:
                return False  # Banner still visible, interaction may have failed
            
            # For accept all, expect groups to be active
            # For reject all, expect minimal groups
            active_groups = consent_data['activeGroups']
            
            if expected_state == ConsentState.ACCEPT_ALL:
                return len(active_groups) > 0  # Should have active groups
            elif expected_state == ConsentState.REJECT_ALL:
                # Should have minimal groups (essential only)
                return len(active_groups) <= 2  # Usually just essential groups
            
            return True
            
        except Exception as e:
            logger.warning(f"Error verifying OneTrust state: {e}")
            return False
    
    async def _verify_cookiebot_state(self, page: Page, expected_state: ConsentState) -> bool:
        """Verify Cookiebot consent state.
        
        Args:
            page: Playwright page object
            expected_state: Expected consent state
            
        Returns:
            True if state matches expectation
        """
        try:
            consent_data = await page.evaluate("""
                () => {
                    if (typeof window.Cookiebot === 'undefined') return null;
                    
                    return {
                        consent: window.Cookiebot.consent || {},
                        configured: window.Cookiebot.configured || false
                    };
                }
            """)
            
            if not consent_data or not consent_data['configured']:
                return False
            
            consent = consent_data['consent']
            
            if expected_state == ConsentState.ACCEPT_ALL:
                # All categories should be true
                return all(consent.get(cat, False) for cat in ['necessary', 'preferences', 'statistics', 'marketing'])
            elif expected_state == ConsentState.REJECT_ALL:
                # Only necessary should be true
                return (consent.get('necessary', False) and 
                       not consent.get('preferences', True) and
                       not consent.get('statistics', True) and
                       not consent.get('marketing', True))
            
            return True
            
        except Exception as e:
            logger.warning(f"Error verifying Cookiebot state: {e}")
            return False
    
    async def _check_modal_dismissed(self, page: Page, platform: CMPPlatform) -> bool:
        """Check if CMP modal has been dismissed.
        
        Args:
            page: Playwright page object
            platform: CMP platform
            
        Returns:
            True if modal appears to be dismissed
        """
        try:
            selectors = self.detector.detection_selectors.get(platform, [])
            
            for selector in selectors:
                element = await page.query_selector(selector)
                if element:
                    is_visible = await element.is_visible()
                    if is_visible:
                        return False  # Modal still visible
            
            return True  # No visible modals found
            
        except Exception as e:
            logger.warning(f"Error checking modal dismissal: {e}")
            return False