"""Unit tests for browser factory."""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from app.audit.capture.browser_factory import (
    BrowserFactory, BrowserConfig, BrowserEngineType,
    create_browser_factory, create_default_factory, create_debug_factory
)


class TestBrowserConfig:
    """Tests for BrowserConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BrowserConfig()
        
        assert config.engine == BrowserEngineType.CHROMIUM
        assert config.headless is True
        assert config.devtools is False
        assert config.viewport == {'width': 1920, 'height': 1080}
        assert config.extra_headers == {}
        assert config.permissions == []
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = BrowserConfig(
            engine=BrowserEngineType.FIREFOX,
            headless=False,
            viewport={'width': 1280, 'height': 720},
            user_agent="Custom UA",
            extra_headers={'Authorization': 'Bearer token'},
            permissions=['geolocation', 'camera']
        )
        
        assert config.engine == BrowserEngineType.FIREFOX
        assert config.headless is False
        assert config.viewport == {'width': 1280, 'height': 720}
        assert config.user_agent == "Custom UA"
        assert config.extra_headers == {'Authorization': 'Bearer token'}
        assert config.permissions == ['geolocation', 'camera']
    
    def test_browser_options_conversion(self):
        """Test conversion to browser launch options."""
        config = BrowserConfig(
            headless=False,
            devtools=True,
            slow_mo=500,
            custom_arg="value"
        )
        
        options = config.to_browser_options()
        
        assert options['headless'] is False
        assert options['devtools'] is True
        assert options['slow_mo'] == 500
        assert options['custom_arg'] == "value"
    
    def test_context_options_conversion(self):
        """Test conversion to context options."""
        config = BrowserConfig(
            viewport={'width': 800, 'height': 600},
            user_agent="Test Agent",
            extra_headers={'Test': 'Header'},
            permissions=['clipboard-read'],
            geolocation={'latitude': 40.7128, 'longitude': -74.0060},
            locale='en-US',
            timezone='America/New_York'
        )
        
        options = config.to_context_options()
        
        assert options['viewport'] == {'width': 800, 'height': 600}
        assert options['user_agent'] == "Test Agent"
        assert options['extra_http_headers'] == {'Test': 'Header'}
        assert options['permissions'] == ['clipboard-read']
        assert options['geolocation'] == {'latitude': 40.7128, 'longitude': -74.0060}
        assert options['locale'] == 'en-US'
        assert options['timezone_id'] == 'America/New_York'


class TestBrowserFactory:
    """Tests for BrowserFactory class."""
    
    @pytest.fixture
    def mock_playwright(self):
        """Mock Playwright instance."""
        with patch('app.audit.capture.browser_factory.async_playwright') as mock_pw:
            playwright_mock = AsyncMock()
            # Fix: Make async_playwright() return an object with async start() method
            async_pw_instance = AsyncMock()
            async_pw_instance.start = AsyncMock(return_value=playwright_mock)
            mock_pw.return_value = async_pw_instance
            
            # Mock browser types
            browser_mock = AsyncMock()
            playwright_mock.chromium.launch.return_value = browser_mock
            playwright_mock.firefox.launch.return_value = browser_mock
            playwright_mock.webkit.launch.return_value = browser_mock
            
            # Mock browser methods
            browser_mock.version = "Test Browser 1.0"
            browser_mock._is_closed = MagicMock(return_value=False)
            
            # Mock context creation
            context_mock = AsyncMock()
            browser_mock.new_context.return_value = context_mock
            
            # Mock page creation
            page_mock = AsyncMock()
            context_mock.new_page.return_value = page_mock
            page_mock.goto.return_value = AsyncMock()
            
            yield {
                'playwright': playwright_mock,
                'browser': browser_mock,
                'context': context_mock,
                'page': page_mock
            }
    
    @pytest.fixture
    def factory(self):
        """Create browser factory for testing."""
        config = BrowserConfig(headless=True)
        return BrowserFactory(config)
    
    @pytest.mark.asyncio
    async def test_factory_start_stop(self, factory, mock_playwright):
        """Test factory start and stop lifecycle."""
        # Test start
        await factory.start()
        
        assert factory.playwright is not None
        assert factory.browser is not None
        assert factory.is_running is True
        
        # Test stop
        await factory.stop()
        
        assert factory.playwright is None
        assert factory.browser is None
        assert factory.context_count == 0
    
    @pytest.mark.asyncio
    async def test_factory_start_already_running(self, factory, mock_playwright):
        """Test starting factory when already running."""
        await factory.start()
        
        # Try to start again - should not raise error
        await factory.start()
        
        assert factory.is_running is True
        
        await factory.stop()
    
    @pytest.mark.asyncio
    async def test_browser_engine_selection(self, mock_playwright):
        """Test different browser engine selection."""
        engines = [
            (BrowserEngineType.CHROMIUM, 'chromium'),
            (BrowserEngineType.FIREFOX, 'firefox'),
            (BrowserEngineType.WEBKIT, 'webkit')
        ]
        
        for engine_type, engine_attr in engines:
            config = BrowserConfig(engine=engine_type)
            factory = BrowserFactory(config)
            
            await factory.start()
            
            # Verify correct browser type was called
            playwright_mock = mock_playwright['playwright']
            engine_mock = getattr(playwright_mock, engine_attr)
            engine_mock.launch.assert_called()
            
            await factory.stop()
    
    @pytest.mark.asyncio
    async def test_create_context(self, factory, mock_playwright):
        """Test browser context creation."""
        await factory.start()
        
        context = await factory.create_context()
        
        assert context is not None
        assert factory.context_count == 1
        
        # Test with overrides
        context2 = await factory.create_context(
            viewport={'width': 1024, 'height': 768}
        )
        
        assert factory.context_count == 2
        
        await factory.stop()
    
    @pytest.mark.asyncio
    async def test_create_context_not_started(self, factory):
        """Test creating context when factory not started."""
        with pytest.raises(RuntimeError, match="Browser factory not started"):
            await factory.create_context()
    
    @pytest.mark.asyncio
    async def test_context_manager(self, factory, mock_playwright):
        """Test context manager for contexts."""
        await factory.start()
        
        initial_count = factory.context_count
        
        async with factory.context() as context:
            assert context is not None
            assert factory.context_count == initial_count + 1
        
        # Context should be closed automatically
        assert factory.context_count == initial_count
        
        await factory.stop()
    
    @pytest.mark.asyncio
    async def test_page_context_manager(self, factory, mock_playwright):
        """Test context manager for pages."""
        await factory.start()
        
        async with factory.page() as page:
            assert page is not None
        
        await factory.stop()
    
    @pytest.mark.asyncio
    async def test_context_limit(self, factory, mock_playwright):
        """Test context count limit handling."""
        await factory.start()
        
        # Set low limit for testing
        factory._max_contexts = 2
        
        # Create contexts up to limit
        context1 = await factory.create_context()
        context2 = await factory.create_context()
        
        # This should still work but log warning
        context3 = await factory.create_context()
        
        assert factory.context_count == 3
        
        await factory.stop()
    
    @pytest.mark.asyncio
    async def test_get_browser_version(self, factory, mock_playwright):
        """Test getting browser version."""
        await factory.start()
        
        version = await factory.get_browser_version()
        assert version == "Test Browser 1.0"
        
        await factory.stop()
        
        # Should return None when not started
        version = await factory.get_browser_version()
        assert version is None
    
    @pytest.mark.asyncio
    async def test_health_check(self, factory, mock_playwright):
        """Test browser health check."""
        await factory.start()
        
        health = await factory.health_check()
        assert health is True
        
        await factory.stop()
        
        # Should return False when not running
        health = await factory.health_check()
        assert health is False
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, factory, mock_playwright):
        """Test health check failure handling."""
        await factory.start()
        
        # Mock page.goto to raise exception
        mock_playwright['page'].goto.side_effect = Exception("Navigation failed")
        
        health = await factory.health_check()
        assert health is False
        
        await factory.stop()
    
    @pytest.mark.asyncio
    async def test_restart_browser(self, factory, mock_playwright):
        """Test browser restart functionality."""
        await factory.start()
        original_config = factory.config
        
        await factory.restart_browser()
        
        # Should still be running with same config
        assert factory.is_running is True
        assert factory.config == original_config
        
        await factory.stop()
    
    @pytest.mark.asyncio
    async def test_factory_repr(self, factory, mock_playwright):
        """Test factory string representation."""
        repr_str = repr(factory)
        assert "BrowserFactory" in repr_str
        assert "chromium" in repr_str
        assert "headless=True" in repr_str
        assert "running=False" in repr_str
        
        await factory.start()
        
        repr_str = repr(factory)
        assert "running=True" in repr_str
        
        await factory.stop()


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_create_browser_factory(self):
        """Test create_browser_factory function."""
        factory = create_browser_factory(
            engine=BrowserEngineType.FIREFOX,
            headless=False,
            viewport={'width': 800, 'height': 600}
        )
        
        assert isinstance(factory, BrowserFactory)
        assert factory.config.engine == BrowserEngineType.FIREFOX
        assert factory.config.headless is False
        assert factory.config.viewport == {'width': 800, 'height': 600}
    
    def test_create_default_factory(self):
        """Test create_default_factory function."""
        factory = create_default_factory()
        
        assert isinstance(factory, BrowserFactory)
        assert factory.config.engine == BrowserEngineType.CHROMIUM
        assert factory.config.headless is True
        assert factory.config.viewport == {'width': 1920, 'height': 1080}
        assert factory.config.ignore_https_errors is True
    
    def test_create_debug_factory(self):
        """Test create_debug_factory function."""
        factory = create_debug_factory()
        
        assert isinstance(factory, BrowserFactory)
        assert factory.config.headless is False
        assert factory.config.devtools is True
        assert factory.config.slow_mo == 500


if __name__ == "__main__":
    pytest.main([__file__])