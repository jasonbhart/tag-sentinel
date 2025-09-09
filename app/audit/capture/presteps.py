"""Pre-steps executor for scripted actions before page capture.

This module provides the PreStepsExecutor class that can execute common
browser actions like clicking, filling forms, waiting for elements, and
running custom JavaScript before the main page capture begins.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError

logger = logging.getLogger(__name__)


class PreStepAction:
    """Base class for pre-step actions."""
    
    def __init__(self, action_type: str, **kwargs):
        """Initialize pre-step action.
        
        Args:
            action_type: Type of action (click, fill, wait, etc.)
            **kwargs: Action-specific parameters
        """
        self.action_type = action_type
        self.params = kwargs
        self.executed_at: Optional[datetime] = None
        self.success: bool = False
        self.error_message: Optional[str] = None
        self.duration_ms: Optional[float] = None
    
    def __repr__(self) -> str:
        """String representation of action."""
        return f"PreStepAction(type={self.action_type}, params={self.params})"


class PreStepsExecutor:
    """Executor for pre-capture browser actions."""
    
    def __init__(self, page: Page, timeout_ms: int = 30000, retry_count: int = 3):
        """Initialize pre-steps executor.
        
        Args:
            page: Playwright page to execute actions on
            timeout_ms: Default timeout for actions in milliseconds
            retry_count: Number of retry attempts for failed actions
        """
        self.page = page
        self.timeout_ms = timeout_ms
        self.retry_count = retry_count
        self.executed_actions: List[PreStepAction] = []
        
        # Action type mapping
        self._action_handlers = {
            'click': self._execute_click,
            'fill': self._execute_fill,
            'select': self._execute_select,
            'wait_for_selector': self._execute_wait_for_selector,
            'wait_for_timeout': self._execute_wait_for_timeout,
            'wait_for_load_state': self._execute_wait_for_load_state,
            'evaluate': self._execute_evaluate,
            'press': self._execute_press,
            'type': self._execute_type,
            'hover': self._execute_hover,
            'scroll': self._execute_scroll,
            'screenshot': self._execute_screenshot,
        }
    
    async def execute_steps(self, steps: List[Dict[str, Any]]) -> bool:
        """Execute a list of pre-step actions.
        
        Args:
            steps: List of action dictionaries
            
        Returns:
            True if all steps executed successfully
        """
        logger.info(f"Executing {len(steps)} pre-steps")
        
        success = True
        
        for i, step in enumerate(steps):
            action_type = step.get('action', '').lower()
            
            if action_type not in self._action_handlers:
                logger.error(f"Unknown action type: {action_type}")
                success = False
                continue
            
            try:
                step_success = await self._execute_action_with_retry(action_type, step)
                if not step_success:
                    success = False
                    
                    # Check if this is a critical step
                    if step.get('critical', False):
                        logger.error(f"Critical step {i+1} failed, stopping execution")
                        break
                        
            except Exception as e:
                logger.error(f"Unexpected error executing step {i+1}: {e}")
                success = False
                
                if step.get('critical', False):
                    break
        
        logger.info(f"Pre-steps execution completed. Success: {success}")
        return success
    
    async def _execute_action_with_retry(self, action_type: str, step: Dict[str, Any]) -> bool:
        """Execute an action with retry logic.
        
        Args:
            action_type: Type of action to execute
            step: Step configuration dictionary
            
        Returns:
            True if action succeeded
        """
        action = PreStepAction(action_type, **step)
        handler = self._action_handlers[action_type]
        
        retry_count = step.get('retry_count', self.retry_count)
        
        for attempt in range(retry_count + 1):
            try:
                start_time = datetime.utcnow()
                action.executed_at = start_time
                
                await handler(step)
                
                # Mark as successful
                end_time = datetime.utcnow()
                action.success = True
                action.duration_ms = (end_time - start_time).total_seconds() * 1000
                
                self.executed_actions.append(action)
                logger.debug(f"Action {action_type} succeeded on attempt {attempt + 1}")
                return True
                
            except Exception as e:
                error_msg = str(e)
                action.error_message = error_msg
                
                if attempt < retry_count:
                    retry_delay = step.get('retry_delay', 1.0) * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Action {action_type} failed on attempt {attempt + 1}: {error_msg}. Retrying in {retry_delay}s")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"Action {action_type} failed after {retry_count + 1} attempts: {error_msg}")
                    
                    # Record failed action
                    action.success = False
                    self.executed_actions.append(action)
                    return False
        
        return False
    
    async def _execute_click(self, step: Dict[str, Any]) -> None:
        """Execute click action.
        
        Args:
            step: Step configuration with 'selector' and optional parameters
        """
        selector = step['selector']
        timeout = step.get('timeout', self.timeout_ms)
        
        # Additional click options
        click_options = {}
        if 'button' in step:
            click_options['button'] = step['button']
        if 'click_count' in step:
            click_options['click_count'] = step['click_count']
        if 'delay' in step:
            click_options['delay'] = step['delay']
        
        await self.page.click(selector, timeout=timeout, **click_options)
        logger.debug(f"Clicked element: {selector}")
    
    async def _execute_fill(self, step: Dict[str, Any]) -> None:
        """Execute fill action.
        
        Args:
            step: Step configuration with 'selector' and 'value'
        """
        selector = step['selector']
        value = step['value']
        timeout = step.get('timeout', self.timeout_ms)
        
        await self.page.fill(selector, value, timeout=timeout)
        logger.debug(f"Filled element {selector} with value")
    
    async def _execute_select(self, step: Dict[str, Any]) -> None:
        """Execute select action.
        
        Args:
            step: Step configuration with 'selector' and 'value'/'label'/'index'
        """
        selector = step['selector']
        timeout = step.get('timeout', self.timeout_ms)
        
        if 'value' in step:
            await self.page.select_option(selector, value=step['value'], timeout=timeout)
        elif 'label' in step:
            await self.page.select_option(selector, label=step['label'], timeout=timeout)
        elif 'index' in step:
            await self.page.select_option(selector, index=step['index'], timeout=timeout)
        else:
            raise ValueError("Select action requires 'value', 'label', or 'index' parameter")
        
        logger.debug(f"Selected option in element: {selector}")
    
    async def _execute_wait_for_selector(self, step: Dict[str, Any]) -> None:
        """Execute wait for selector action.
        
        Args:
            step: Step configuration with 'selector' and optional 'state'
        """
        selector = step['selector']
        timeout = step.get('timeout', self.timeout_ms)
        state = step.get('state', 'visible')
        
        await self.page.wait_for_selector(selector, state=state, timeout=timeout)
        logger.debug(f"Waited for selector: {selector} (state: {state})")
    
    async def _execute_wait_for_timeout(self, step: Dict[str, Any]) -> None:
        """Execute wait for timeout action.
        
        Args:
            step: Step configuration with 'timeout' in milliseconds
        """
        timeout = step['timeout']
        await self.page.wait_for_timeout(timeout)
        logger.debug(f"Waited for timeout: {timeout}ms")
    
    async def _execute_wait_for_load_state(self, step: Dict[str, Any]) -> None:
        """Execute wait for load state action.
        
        Args:
            step: Step configuration with optional 'state'
        """
        state = step.get('state', 'load')
        timeout = step.get('timeout', self.timeout_ms)
        
        await self.page.wait_for_load_state(state, timeout=timeout)
        logger.debug(f"Waited for load state: {state}")
    
    async def _execute_evaluate(self, step: Dict[str, Any]) -> None:
        """Execute JavaScript evaluation.
        
        Args:
            step: Step configuration with 'script' and optional 'args'
        """
        script = step['script']
        args = step.get('args', [])
        
        result = await self.page.evaluate(script, args)
        logger.debug(f"Executed JavaScript, result: {result}")
    
    async def _execute_press(self, step: Dict[str, Any]) -> None:
        """Execute key press action.
        
        Args:
            step: Step configuration with 'key' and optional 'selector'
        """
        key = step['key']
        selector = step.get('selector')
        timeout = step.get('timeout', self.timeout_ms)
        
        if selector:
            await self.page.press(selector, key, timeout=timeout)
        else:
            await self.page.keyboard.press(key)
        
        logger.debug(f"Pressed key: {key}")
    
    async def _execute_type(self, step: Dict[str, Any]) -> None:
        """Execute typing action.
        
        Args:
            step: Step configuration with 'text' and optional 'selector'
        """
        text = step['text']
        selector = step.get('selector')
        delay = step.get('delay', 0)
        timeout = step.get('timeout', self.timeout_ms)
        
        if selector:
            await self.page.type(selector, text, delay=delay, timeout=timeout)
        else:
            await self.page.keyboard.type(text, delay=delay)
        
        logger.debug(f"Typed text: {len(text)} characters")
    
    async def _execute_hover(self, step: Dict[str, Any]) -> None:
        """Execute hover action.
        
        Args:
            step: Step configuration with 'selector'
        """
        selector = step['selector']
        timeout = step.get('timeout', self.timeout_ms)
        
        await self.page.hover(selector, timeout=timeout)
        logger.debug(f"Hovered over element: {selector}")
    
    async def _execute_scroll(self, step: Dict[str, Any]) -> None:
        """Execute scroll action.
        
        Args:
            step: Step configuration with optional 'selector', 'x', 'y'
        """
        selector = step.get('selector')
        x = step.get('x', 0)
        y = step.get('y', 0)
        
        if selector:
            element = await self.page.query_selector(selector)
            if element:
                await element.scroll_into_view_if_needed()
        else:
            await self.page.evaluate(f'window.scrollBy({x}, {y})')
        
        logger.debug(f"Scrolled to position: x={x}, y={y}")
    
    async def _execute_screenshot(self, step: Dict[str, Any]) -> None:
        """Execute screenshot action.
        
        Args:
            step: Step configuration with optional 'path' and 'full_page'
        """
        path = step.get('path')
        full_page = step.get('full_page', False)
        
        screenshot_options = {'full_page': full_page}
        if path:
            screenshot_options['path'] = path
        
        await self.page.screenshot(**screenshot_options)
        logger.debug(f"Took screenshot: {path or 'buffer'}")
    
    def get_executed_actions(self) -> List[PreStepAction]:
        """Get list of executed actions.
        
        Returns:
            List of PreStepAction objects
        """
        return self.executed_actions.copy()
    
    def get_successful_actions(self) -> List[PreStepAction]:
        """Get list of successful actions.
        
        Returns:
            List of successful PreStepAction objects
        """
        return [action for action in self.executed_actions if action.success]
    
    def get_failed_actions(self) -> List[PreStepAction]:
        """Get list of failed actions.
        
        Returns:
            List of failed PreStepAction objects
        """
        return [action for action in self.executed_actions if not action.success]
    
    def get_total_duration_ms(self) -> float:
        """Get total execution duration in milliseconds.
        
        Returns:
            Total duration of all actions
        """
        return sum(action.duration_ms or 0 for action in self.executed_actions)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics.
        
        Returns:
            Dictionary with execution statistics
        """
        successful = self.get_successful_actions()
        failed = self.get_failed_actions()
        
        return {
            'total_actions': len(self.executed_actions),
            'successful_actions': len(successful),
            'failed_actions': len(failed),
            'success_rate': len(successful) / len(self.executed_actions) * 100 if self.executed_actions else 0,
            'total_duration_ms': self.get_total_duration_ms(),
            'average_duration_ms': self.get_total_duration_ms() / len(self.executed_actions) if self.executed_actions else 0,
        }
    
    def clear(self) -> None:
        """Clear executed actions history."""
        self.executed_actions.clear()
        logger.debug("Pre-steps executor cleared")
    
    def __repr__(self) -> str:
        """String representation of executor."""
        stats = self.get_stats()
        return (
            f"PreStepsExecutor(actions={stats['total_actions']}, "
            f"success_rate={stats['success_rate']:.1f}%)"
        )


# Convenience functions for common pre-step scenarios

def create_login_steps(username_selector: str, password_selector: str, 
                      submit_selector: str, username: str, password: str) -> List[Dict[str, Any]]:
    """Create common login steps.
    
    Args:
        username_selector: CSS selector for username field
        password_selector: CSS selector for password field  
        submit_selector: CSS selector for submit button
        username: Username to fill
        password: Password to fill
        
    Returns:
        List of step dictionaries for login flow
    """
    return [
        {
            'action': 'wait_for_selector',
            'selector': username_selector,
            'critical': True
        },
        {
            'action': 'fill',
            'selector': username_selector,
            'value': username,
            'critical': True
        },
        {
            'action': 'fill',
            'selector': password_selector,
            'value': password,
            'critical': True
        },
        {
            'action': 'click',
            'selector': submit_selector,
            'critical': True
        },
        {
            'action': 'wait_for_load_state',
            'state': 'networkidle'
        }
    ]


def create_cookie_consent_steps(accept_selector: str) -> List[Dict[str, Any]]:
    """Create cookie consent acceptance steps.
    
    Args:
        accept_selector: CSS selector for accept button
        
    Returns:
        List of step dictionaries for cookie consent
    """
    return [
        {
            'action': 'wait_for_selector',
            'selector': accept_selector,
            'timeout': 5000  # Short timeout since not all sites have consent
        },
        {
            'action': 'click',
            'selector': accept_selector
        },
        {
            'action': 'wait_for_timeout',
            'timeout': 1000  # Wait for consent to process
        }
    ]


def create_page_load_steps(wait_selector: Optional[str] = None) -> List[Dict[str, Any]]:
    """Create steps to ensure page is fully loaded.
    
    Args:
        wait_selector: Optional selector to wait for
        
    Returns:
        List of step dictionaries for page loading
    """
    steps = [
        {
            'action': 'wait_for_load_state',
            'state': 'load'
        },
        {
            'action': 'wait_for_load_state', 
            'state': 'networkidle'
        }
    ]
    
    if wait_selector:
        steps.append({
            'action': 'wait_for_selector',
            'selector': wait_selector
        })
    
    return steps