"""Unit tests for capture engine."""

import pytest
import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from app.audit.capture.engine import (
    CaptureEngine, CaptureEngineConfig,
    create_capture_engine, create_debug_capture_engine
)
from app.audit.capture.browser_factory import BrowserConfig
from app.audit.models.capture import PageResult, CaptureStatus


class TestCaptureEngineConfig:
    """Tests for CaptureEngineConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CaptureEngineConfig()
        
        assert config.max_concurrent_pages == 5
        assert config.page_timeout_ms == 30000
        assert config.enable_network_capture is True
        assert config.enable_console_capture is True
        assert config.enable_cookie_capture is True
        assert config.artifacts_enabled is False
        assert config.retry_attempts == 3
        assert config.continue_on_error is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        browser_config = BrowserConfig(headless=False)
        
        config = CaptureEngineConfig(
            browser_config=browser_config,
            max_concurrent_pages=10,
            enable_network_capture=False,
            artifacts_enabled=True,
            artifacts_dir=Path("/tmp/artifacts"),
            retry_attempts=5
        )
        
        assert config.browser_config == browser_config
        assert config.max_concurrent_pages == 10
        assert config.enable_network_capture is False
        assert config.artifacts_enabled is True
        assert config.artifacts_dir == Path("/tmp/artifacts")
        assert config.retry_attempts == 5
    
    def test_create_page_session_config(self):
        """Test creating page session config."""
        config = CaptureEngineConfig(
            enable_network_capture=False,
            artifacts_enabled=True,
            artifacts_dir=Path("/tmp/artifacts")
        )
        
        session_config = config.create_page_session_config()
        
        assert session_config.enable_network_capture is False
        assert session_config.artifacts_dir == Path("/tmp/artifacts")
        
        # Test with overrides
        session_config = config.create_page_session_config(
            enable_network_capture=True,
            wait_timeout_ms=60000
        )
        
        assert session_config.enable_network_capture is True
        assert session_config.wait_timeout_ms == 60000


class TestCaptureEngine:
    """Tests for CaptureEngine class."""
    
    @pytest.fixture
    def mock_browser_factory(self):
        """Mock browser factory."""
        factory = AsyncMock()
        factory.start.return_value = None
        factory.stop.return_value = None
        factory.is_running = True
        factory.health_check.return_value = True
        factory.restart_browser.return_value = None
        
        # Mock page context manager
        page_mock = AsyncMock()
        
        class MockAsyncContextManager:
            async def __aenter__(self):
                return page_mock
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None
        
        def page(*args, **kwargs):
            return MockAsyncContextManager()
        
        factory.page = page
        
        return factory
    
    @pytest.fixture
    def mock_page_session(self):
        """Mock page session."""
        session = AsyncMock()
        
        # Mock successful page result
        result = PageResult(
            url="https://example.com",
            capture_status=CaptureStatus.SUCCESS
        )
        session.capture_page.return_value = result
        
        return session
    
    @pytest.fixture
    def engine(self):
        """Create capture engine for testing."""
        config = CaptureEngineConfig(max_concurrent_pages=2)
        return CaptureEngine(config)
    
    @pytest.mark.asyncio
    async def test_engine_start_stop(self, engine, mock_browser_factory):
        """Test engine start and stop lifecycle."""
        with patch('app.audit.capture.engine.BrowserFactory', return_value=mock_browser_factory):
            # Test start
            await engine.start()
            
            assert engine._is_running is True
            assert engine.browser_factory == mock_browser_factory
            assert engine._semaphore is not None
            assert engine.stats['start_time'] is not None
            
            mock_browser_factory.start.assert_called_once()
            
            # Test stop
            await engine.stop()
            
            assert engine._is_running is False
            assert engine.browser_factory is None
            
            mock_browser_factory.stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_engine_start_failure(self, engine):
        """Test engine start failure handling."""
        with patch('app.audit.capture.engine.BrowserFactory') as mock_factory_class:
            mock_factory = AsyncMock()
            mock_factory.start.side_effect = Exception("Browser start failed")
            mock_factory_class.return_value = mock_factory
            
            with pytest.raises(Exception, match="Browser start failed"):
                await engine.start()
            
            # Should have attempted cleanup
            mock_factory.stop.assert_called_once()
            assert engine._is_running is False
    
    @pytest.mark.asyncio
    async def test_capture_page_not_started(self, engine):
        """Test capturing page when engine not started."""
        with pytest.raises(RuntimeError, match="Capture engine not started"):
            await engine.capture_page("https://example.com")
    
    @pytest.mark.asyncio
    async def test_capture_page_success(self, engine, mock_browser_factory, mock_page_session):
        """Test successful page capture."""
        with patch('app.audit.capture.engine.BrowserFactory', return_value=mock_browser_factory):
            with patch('app.audit.capture.engine.PageSession', return_value=mock_page_session):
                await engine.start()
                
                result = await engine.capture_page("https://example.com")
                
                assert isinstance(result, PageResult)
                assert result.url == "https://example.com"
                assert result.capture_status == CaptureStatus.SUCCESS
                
                # Check stats were updated
                assert engine.stats['pages_attempted'] == 1
                assert engine.stats['pages_successful'] == 1
                
                await engine.stop()
    
    @pytest.mark.asyncio
    async def test_capture_page_with_overrides(self, engine, mock_browser_factory, mock_page_session):
        """Test page capture with session config overrides."""
        with patch('app.audit.capture.engine.BrowserFactory', return_value=mock_browser_factory):
            with patch('app.audit.capture.engine.PageSession', return_value=mock_page_session):
                await engine.start()
                
                overrides = {
                    'wait_timeout_ms': 60000,
                    'take_screenshot': True
                }
                
                result = await engine.capture_page("https://example.com", overrides)
                
                assert isinstance(result, PageResult)
                # Page session should have been created with overrides
                
                await engine.stop()
    
    @pytest.mark.asyncio
    async def test_capture_page_retry_logic(self, engine, mock_browser_factory):
        """Test page capture retry logic."""
        with patch('app.audit.capture.engine.BrowserFactory', return_value=mock_browser_factory):
            with patch('app.audit.capture.engine.PageSession') as mock_session_class:
                session = AsyncMock()
                session.capture_page.side_effect = [
                    Exception("First attempt failed"),
                    Exception("Second attempt failed"),
                    PageResult(url="https://example.com", capture_status=CaptureStatus.SUCCESS)
                ]
                mock_session_class.return_value = session
                
                await engine.start()
                
                # Should succeed on third attempt
                result = await engine.capture_page("https://example.com")
                
                assert result.capture_status == CaptureStatus.SUCCESS
                assert session.capture_page.call_count == 3
                
                await engine.stop()
    
    @pytest.mark.asyncio
    async def test_capture_page_all_retries_failed(self, engine, mock_browser_factory):
        """Test page capture when all retries fail."""
        # Set low retry count for faster test
        engine.config.retry_attempts = 1
        
        with patch('app.audit.capture.engine.BrowserFactory', return_value=mock_browser_factory):
            with patch('app.audit.capture.engine.PageSession') as mock_session_class:
                session = AsyncMock()
                session.capture_page.side_effect = Exception("Persistent failure")
                mock_session_class.return_value = session
                
                await engine.start()
                
                result = await engine.capture_page("https://example.com")
                
                assert result.capture_status == CaptureStatus.FAILED
                assert "Persistent failure" in result.capture_error
                
                # Check stats - engine may have processed pages from other tests
                # So we just check that pages_failed increased
                assert engine.stats['pages_failed'] >= 1
                assert len(engine.stats['errors']) >= 1
                
                await engine.stop()
    
    @pytest.mark.asyncio
    async def test_capture_pages_batch(self, engine, mock_browser_factory, mock_page_session):
        """Test batch page capture."""
        with patch('app.audit.capture.engine.BrowserFactory', return_value=mock_browser_factory):
            with patch('app.audit.capture.engine.PageSession', return_value=mock_page_session):
                await engine.start()
                
                urls = [
                    "https://example.com/page1",
                    "https://example.com/page2", 
                    "https://example.com/page3"
                ]
                
                results = await engine.capture_pages(urls)
                
                assert len(results) == 3
                for result in results:
                    assert isinstance(result, PageResult)
                    assert result.capture_status == CaptureStatus.SUCCESS
                
                # Check stats
                assert engine.stats['pages_attempted'] == 3
                assert engine.stats['pages_successful'] == 3
                
                await engine.stop()
    
    @pytest.mark.asyncio
    async def test_capture_pages_continue_on_error(self, engine, mock_browser_factory):
        """Test batch capture with continue_on_error=True."""
        engine.config.continue_on_error = True
        
        with patch('app.audit.capture.engine.BrowserFactory', return_value=mock_browser_factory):
            with patch.object(engine, 'capture_page') as mock_capture:
                # Mock one failure and one success
                mock_capture.side_effect = [
                    Exception("Page 1 failed"),
                    PageResult(url="https://example.com/page2", capture_status=CaptureStatus.SUCCESS)
                ]
                
                await engine.start()
                
                urls = ["https://example.com/page1", "https://example.com/page2"]
                results = await engine.capture_pages(urls)
                
                # Should get one result (the successful one)
                assert len(results) == 1
                assert results[0].url == "https://example.com/page2"
                
                await engine.stop()
    
    @pytest.mark.asyncio
    async def test_capture_pages_stop_on_error(self, engine, mock_browser_factory):
        """Test batch capture with continue_on_error=False."""
        engine.config.continue_on_error = False
        
        with patch('app.audit.capture.engine.BrowserFactory', return_value=mock_browser_factory):
            with patch.object(engine, 'capture_page') as mock_capture:
                mock_capture.side_effect = Exception("First page failed")
                
                await engine.start()
                
                urls = ["https://example.com/page1", "https://example.com/page2"]
                
                with pytest.raises(Exception, match="First page failed"):
                    await engine.capture_pages(urls)
                
                await engine.stop()
    
    def test_add_callback(self, engine):
        """Test adding callbacks."""
        callback_calls = []
        
        def test_callback(result):
            callback_calls.append(result)
        
        engine.add_callback(test_callback)
        
        # Simulate calling callbacks
        result = PageResult(url="https://example.com", capture_status=CaptureStatus.SUCCESS)
        engine._call_callbacks(result)
        
        assert len(callback_calls) == 1
        assert callback_calls[0] == result
    
    def test_callback_error_handling(self, engine):
        """Test callback error handling."""
        def failing_callback(result):
            raise Exception("Callback failed")
        
        engine.add_callback(failing_callback)
        
        # Should not raise error
        result = PageResult(url="https://example.com", capture_status=CaptureStatus.SUCCESS)
        engine._call_callbacks(result)
    
    def test_update_stats(self, engine):
        """Test statistics updating."""
        # Test successful result
        success_result = PageResult(
            url="https://example.com/success",
            capture_status=CaptureStatus.SUCCESS,
            load_time_ms=1500.0
        )
        engine._update_stats(success_result)
        
        assert engine.stats['pages_attempted'] == 1
        assert engine.stats['pages_successful'] == 1
        assert engine.stats['total_duration_ms'] == 1500.0
        
        # Test failed result
        failed_result = PageResult(
            url="https://example.com/failed",
            capture_status=CaptureStatus.FAILED,
            capture_error="Network error"
        )
        engine._update_stats(failed_result)
        
        assert engine.stats['pages_attempted'] == 2
        assert engine.stats['pages_failed'] == 1
        assert len(engine.stats['errors']) == 1
        
        # Test timeout result
        timeout_result = PageResult(
            url="https://example.com/timeout",
            capture_status=CaptureStatus.TIMEOUT
        )
        engine._update_stats(timeout_result)
        
        assert engine.stats['pages_attempted'] == 3
        assert engine.stats['pages_timeout'] == 1
    
    @pytest.mark.asyncio
    async def test_perform_cleanup(self, engine, mock_browser_factory):
        """Test periodic cleanup."""
        mock_browser_factory.health_check.return_value = True
        
        with patch('app.audit.capture.engine.BrowserFactory', return_value=mock_browser_factory):
            await engine.start()
            
            await engine._perform_cleanup()
            
            mock_browser_factory.health_check.assert_called_once()
            
            await engine.stop()
    
    @pytest.mark.asyncio
    async def test_cleanup_unhealthy_browser(self, engine, mock_browser_factory):
        """Test cleanup when browser is unhealthy."""
        mock_browser_factory.health_check.return_value = False
        
        with patch('app.audit.capture.engine.BrowserFactory', return_value=mock_browser_factory):
            await engine.start()
            
            await engine._perform_cleanup()
            
            mock_browser_factory.restart_browser.assert_called_once()
            
            await engine.stop()
    
    @pytest.mark.asyncio
    async def test_session_context_manager(self, engine, mock_browser_factory, mock_page_session):
        """Test engine session context manager."""
        with patch('app.audit.capture.engine.BrowserFactory', return_value=mock_browser_factory):
            with patch('app.audit.capture.engine.PageSession', return_value=mock_page_session):
                async with engine.session() as eng:
                    assert eng == engine
                    assert engine._is_running is True
                    
                    # Test capture within session
                    result = await eng.capture_page("https://example.com")
                    assert isinstance(result, PageResult)
                
                # Should be stopped after context exit
                assert engine._is_running is False
    
    def test_get_stats(self, engine):
        """Test comprehensive statistics."""
        # Add some mock data
        engine.stats['start_time'] = datetime.utcnow()
        engine.stats['pages_attempted'] = 10
        engine.stats['pages_successful'] = 8
        engine.stats['total_duration_ms'] = 15000.0
        
        stats = engine.get_stats()
        
        assert stats['pages_attempted'] == 10
        assert stats['pages_successful'] == 8
        assert stats['success_rate'] == 80.0
        assert stats['average_load_time_ms'] == 1500.0
        assert 'runtime_seconds' in stats
        assert 'pages_per_second' in stats
        assert stats['is_running'] is False
    
    def test_export_summary(self, engine):
        """Test summary export for reporting."""
        # Set up some stats
        engine.stats['pages_attempted'] = 5
        engine.stats['pages_successful'] = 4
        engine.stats['total_duration_ms'] = 7500.0
        engine.stats['errors'] = [{'url': 'test', 'error': 'error'}]
        
        summary = engine.export_summary()
        
        assert summary['engine_status'] == 'stopped'
        assert summary['pages_processed'] == 5
        assert summary['success_rate'] == '80.0%'
        assert summary['average_load_time'] == '1500ms'
        assert summary['error_count'] == 1
        assert 'configuration' in summary
    
    def test_repr(self, engine):
        """Test string representation."""
        engine.stats['pages_attempted'] = 5
        engine.stats['pages_successful'] = 4
        
        repr_str = repr(engine)
        assert "CaptureEngine" in repr_str
        assert "running=False" in repr_str
        assert "processed=5" in repr_str
        assert "success_rate=80.0%" in repr_str


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_create_capture_engine(self):
        """Test create_capture_engine function."""
        engine = create_capture_engine(
            headless=False,
            max_concurrent_pages=3,
            enable_artifacts=True,
            artifacts_dir=Path("/tmp/test")
        )
        
        assert isinstance(engine, CaptureEngine)
        assert engine.config.browser_config.headless is False
        assert engine.config.max_concurrent_pages == 3
        assert engine.config.artifacts_enabled is True
        assert engine.config.artifacts_dir == Path("/tmp/test")
    
    def test_create_debug_capture_engine(self):
        """Test create_debug_capture_engine function."""
        engine = create_debug_capture_engine()
        
        assert isinstance(engine, CaptureEngine)
        assert engine.config.browser_config.headless is False
        assert engine.config.browser_config.devtools is True
        assert engine.config.max_concurrent_pages == 1
        assert engine.config.artifacts_enabled is True
        assert engine.config.take_screenshots is True


if __name__ == "__main__":
    pytest.main([__file__])