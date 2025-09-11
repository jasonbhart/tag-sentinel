"""Unit tests for DataLayer service orchestration."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.audit.datalayer.service import DataLayerService
from app.audit.datalayer.models import (
    DataLayerSnapshot,
    ValidationIssue,
    ValidationSeverity,
    DLContext,
    DLResult
)
from app.audit.datalayer.config import DataLayerConfig


class TestDataLayerService:
    """Test cases for DataLayerService class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = DataLayerConfig(enabled=True)
        self.service = DataLayerService(self.config)
    
    def test_service_initialization(self):
        """Test service initialization."""
        assert self.service.config == self.config
        assert self.service.snapshotter is not None
        assert self.service.redactor is not None
        assert self.service.validator is not None
        assert self.service.aggregator is None  # Not initialized until needed
    
    @pytest.mark.asyncio
    async def test_capture_and_validate_success(self):
        """Test successful capture and validation."""
        # Mock page
        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.evaluate.return_value = {"page": "home", "user_id": "123"}
        
        context = DLContext(url="https://example.com")
        
        # Mock schema for validation
        schema = {
            "type": "object",
            "properties": {
                "page": {"type": "string"},
                "user_id": {"type": "string"}
            },
            "required": ["page"]
        }
        
        result = await self.service.capture_and_validate(mock_page, context, schema)
        
        assert result.success
        assert result.snapshot.has_data
        assert result.snapshot.url == "https://example.com"
        assert result.snapshot.processed_data["page"] == "home"
        assert len(result.issues) == 0  # Valid data, no issues
    
    @pytest.mark.asyncio
    async def test_capture_and_validate_with_validation_errors(self):
        """Test capture with validation errors."""
        # Mock page with invalid data
        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.evaluate.return_value = {"invalid_field": "value"}
        
        context = DLContext(url="https://example.com")
        
        # Schema that requires 'page' field
        schema = {
            "type": "object",
            "properties": {"page": {"type": "string"}},
            "required": ["page"]
        }
        
        result = await self.service.capture_and_validate(mock_page, context, schema)
        
        assert not result.success  # Should fail due to validation errors
        assert result.snapshot.has_data
        assert len(result.issues) > 0
        assert any(issue.severity == ValidationSeverity.ERROR for issue in result.issues)
    
    @pytest.mark.asyncio
    async def test_capture_and_validate_with_redaction(self):
        """Test capture with sensitive data redaction."""
        # Mock page with sensitive data
        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.evaluate.return_value = {
            "page": "home",
            "user_email": "user@example.com",
            "phone": "555-123-4567"
        }
        
        context = DLContext(url="https://example.com")
        
        # Configure redaction
        redaction_paths = ["/user_email", "/phone"]
        
        result = await self.service.capture_and_validate(
            mock_page, context, redaction_paths=redaction_paths
        )
        
        assert result.success
        assert result.snapshot.has_data
        
        # Sensitive data should be redacted
        processed_data = result.snapshot.processed_data
        assert processed_data["user_email"] != "user@example.com"
        assert processed_data["phone"] != "555-123-4567"
        assert processed_data["page"] == "home"  # Non-sensitive data unchanged
        
        # Should have redaction audit trail
        assert len(result.redaction_audit) == 2
    
    @pytest.mark.asyncio
    async def test_capture_and_validate_missing_datalayer(self):
        """Test capture when dataLayer is missing."""
        # Mock page with no dataLayer
        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.evaluate.return_value = None
        
        context = DLContext(url="https://example.com")
        
        result = await self.service.capture_and_validate(mock_page, context)
        
        assert not result.success
        assert not result.snapshot.has_data
        assert result.snapshot.capture_error is not None
    
    @pytest.mark.asyncio
    async def test_capture_and_validate_timeout(self):
        """Test capture with timeout error."""
        # Mock page that times out
        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.evaluate.side_effect = asyncio.TimeoutError("Timeout")
        
        context = DLContext(url="https://example.com")
        
        result = await self.service.capture_and_validate(mock_page, context)
        
        assert not result.success
        assert not result.snapshot.has_data
        assert "timeout" in result.snapshot.capture_error.lower()
    
    @pytest.mark.asyncio
    async def test_process_multiple_pages(self):
        """Test processing multiple pages."""
        # Create mock page contexts
        contexts = [
            DLContext(url="https://example.com/page1"),
            DLContext(url="https://example.com/page2"),
            DLContext(url="https://example.com/page3")
        ]
        
        # Mock successful processing
        with patch.object(self.service, 'capture_and_validate') as mock_capture:
            mock_results = []
            for i, context in enumerate(contexts):
                snapshot = DataLayerSnapshot(
                    url=context.url,
                    raw_data={"page": f"page{i+1}"},
                    processed_data={"page": f"page{i+1}"}
                )
                result = DLResult(context=context, snapshot=snapshot)
                mock_results.append(result)
            
            mock_capture.side_effect = mock_results
            
            # Process pages (mock pages)
            mock_pages = [Mock() for _ in contexts]
            results = await self.service.process_multiple_pages(mock_pages, contexts)
            
            assert len(results) == 3
            assert all(result.success for result in results)
            assert mock_capture.call_count == 3
    
    @pytest.mark.asyncio
    async def test_process_multiple_pages_with_concurrency(self):
        """Test concurrent processing of multiple pages."""
        # Create many contexts to test concurrency
        contexts = [DLContext(url=f"https://example.com/page{i}") for i in range(10)]
        
        # Mock processing with delay to test concurrency
        async def mock_capture(page, context, *args, **kwargs):
            await asyncio.sleep(0.01)  # Small delay
            snapshot = DataLayerSnapshot(
                url=context.url,
                raw_data={"page": "test"},
                processed_data={"page": "test"}
            )
            return DLResult(context=context, snapshot=snapshot)
        
        with patch.object(self.service, 'capture_and_validate', side_effect=mock_capture):
            mock_pages = [Mock() for _ in contexts]
            
            import time
            start_time = time.time()
            results = await self.service.process_multiple_pages(mock_pages, contexts, max_concurrency=5)
            end_time = time.time()
            
            assert len(results) == 10
            # With concurrency, should be faster than sequential processing
            assert end_time - start_time < 0.5  # Should complete quickly with concurrency
    
    @pytest.mark.asyncio
    async def test_batch_process_with_aggregation(self):
        """Test batch processing with aggregation."""
        # Create test data
        contexts = [DLContext(url=f"https://example.com/page{i}") for i in range(5)]
        
        # Mock processing results
        mock_results = []
        for i, context in enumerate(contexts):
            snapshot = DataLayerSnapshot(
                url=context.url,
                raw_data={"page": f"page{i}", "user_id": "123"},
                processed_data={"page": f"page{i}", "user_id": "123"}
            )
            result = DLResult(context=context, snapshot=snapshot)
            mock_results.append(result)
        
        with patch.object(self.service, 'process_multiple_pages', return_value=mock_results):
            mock_pages = [Mock() for _ in contexts]
            
            aggregate = await self.service.batch_process_with_aggregation(
                pages=mock_pages,
                contexts=contexts,
                run_id="test-batch-123"
            )
            
            assert aggregate.run_id == "test-batch-123"
            assert aggregate.total_pages == 5
            assert aggregate.successful_captures == 5
            assert aggregate.success_rate == 100.0
    
    def test_get_processing_stats(self):
        """Test getting processing statistics."""
        stats = self.service.get_processing_stats()
        
        assert "total_requests" in stats
        assert "successful_requests" in stats
        assert "failed_requests" in stats
        assert "average_processing_time" in stats
        
        # Initial stats should be empty
        assert stats["total_requests"] == 0
        assert stats["successful_requests"] == 0
        assert stats["failed_requests"] == 0
    
    def test_get_health_status(self):
        """Test getting health status."""
        health = self.service.get_health_status()
        
        assert "service_name" in health
        assert "status" in health
        assert "components" in health
        assert "uptime" in health
        
        # Should be healthy initially
        assert health["status"] == "healthy"
        
        # Should have component health
        components = health["components"]
        assert "snapshotter" in components
        assert "redactor" in components
        assert "validator" in components
    
    def test_reset_stats(self):
        """Test resetting processing statistics."""
        # Simulate some processing (mock internal counters)
        self.service._stats["total_requests"] = 10
        self.service._stats["successful_requests"] = 8
        self.service._stats["failed_requests"] = 2
        
        self.service.reset_stats()
        
        stats = self.service.get_processing_stats()
        assert stats["total_requests"] == 0
        assert stats["successful_requests"] == 0
        assert stats["failed_requests"] == 0
    
    @pytest.mark.asyncio
    async def test_service_with_disabled_config(self):
        """Test service behavior when disabled in config."""
        disabled_config = DataLayerConfig(enabled=False)
        disabled_service = DataLayerService(disabled_config)
        
        mock_page = AsyncMock()
        context = DLContext(url="https://example.com")
        
        result = await disabled_service.capture_and_validate(mock_page, context)
        
        # Should return a result indicating service is disabled
        assert not result.success
        assert not result.snapshot.has_data
    
    def test_service_configuration_update(self):
        """Test updating service configuration."""
        new_config = DataLayerConfig(
            enabled=True,
            capture_timeout=20.0,
            max_depth=100
        )
        
        self.service.update_config(new_config)
        
        assert self.service.config == new_config
        assert self.service.snapshotter.config == new_config
        assert self.service.redactor.config.enabled == new_config.redaction.enabled
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        # Mock page that throws an error
        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.evaluate.side_effect = Exception("Unexpected error")
        
        context = DLContext(url="https://example.com")
        
        # Should handle error gracefully
        result = await self.service.capture_and_validate(mock_page, context)
        
        assert not result.success
        assert result.snapshot.capture_error is not None
        
        # Service should still be operational for next request
        health = self.service.get_health_status()
        assert health["status"] in ["healthy", "degraded"]  # Should not be "unhealthy"
    
    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self):
        """Test handling multiple concurrent requests."""
        # Create multiple concurrent requests
        contexts = [DLContext(url=f"https://example.com/concurrent{i}") for i in range(5)]
        
        # Mock pages
        mock_pages = []
        for i, context in enumerate(contexts):
            mock_page = AsyncMock()
            mock_page.url = context.url
            mock_page.evaluate.return_value = {"page": f"concurrent{i}"}
            mock_pages.append(mock_page)
        
        # Process all requests concurrently
        tasks = []
        for page, context in zip(mock_pages, contexts):
            task = asyncio.create_task(self.service.capture_and_validate(page, context))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        assert all(result.success for result in results)
        
        # Each result should have correct URL
        for i, result in enumerate(results):
            assert result.context.url == f"https://example.com/concurrent{i}"
    
    def test_memory_usage_monitoring(self):
        """Test memory usage monitoring."""
        health = self.service.get_health_status()
        
        # Should include memory information
        assert "memory_usage" in health
        assert isinstance(health["memory_usage"], (int, float))
        assert health["memory_usage"] >= 0
    
    def test_service_metrics_collection(self):
        """Test collection of service metrics."""
        # Simulate some processing
        self.service._record_request(success=True, processing_time=0.1)
        self.service._record_request(success=True, processing_time=0.2)
        self.service._record_request(success=False, processing_time=0.05)
        
        stats = self.service.get_processing_stats()
        
        assert stats["total_requests"] == 3
        assert stats["successful_requests"] == 2
        assert stats["failed_requests"] == 1
        assert stats["success_rate"] == pytest.approx(66.67, rel=1e-2)
        assert stats["average_processing_time"] > 0
    
    @pytest.mark.asyncio
    async def test_service_shutdown_cleanup(self):
        """Test service cleanup on shutdown."""
        # Create service with some state
        contexts = [DLContext(url="https://example.com/test")]
        
        with patch.object(self.service, 'process_multiple_pages') as mock_process:
            mock_process.return_value = []
            
            # Start batch processing
            mock_pages = [Mock()]
            await self.service.batch_process_with_aggregation(mock_pages, contexts, "test-run")
            
            # Service should have aggregator initialized
            assert self.service.aggregator is not None
            
            # Cleanup should reset state appropriately
            await self.service.cleanup()
            
            # Verify cleanup (exact behavior depends on implementation)
            # At minimum, should not raise exceptions


class TestServiceIntegration:
    """Integration tests for service components working together."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.config = DataLayerConfig(
            enabled=True,
            capture_timeout=5.0,
            redaction={"enabled": True, "default_action": "HASH"},
            validation={"enabled": True, "strict_mode": False}
        )
        self.service = DataLayerService(self.config)
    
    @pytest.mark.asyncio
    async def test_end_to_end_processing(self):
        """Test complete end-to-end processing pipeline."""
        # Mock page with realistic dataLayer
        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.evaluate.return_value = {
            "page": "home",
            "user_id": "user123",
            "email": "user@example.com",
            "products": ["item1", "item2"],
            "event": "page_view"
        }
        
        context = DLContext(url="https://example.com", page_title="Home Page")
        
        # Schema for validation
        schema = {
            "type": "object",
            "properties": {
                "page": {"type": "string"},
                "user_id": {"type": "string"},
                "products": {"type": "array"},
                "email": {"type": "string", "format": "email"}
            },
            "required": ["page"]
        }
        
        # Redaction paths
        redaction_paths = ["/email"]
        
        result = await self.service.capture_and_validate(
            mock_page, context, schema, redaction_paths
        )
        
        # Verify complete processing
        assert result.success
        assert result.snapshot.has_data
        assert result.snapshot.url == "https://example.com"
        
        # Data should be captured
        assert result.snapshot.processed_data["page"] == "home"
        assert result.snapshot.processed_data["user_id"] == "user123"
        
        # Email should be redacted
        assert result.snapshot.processed_data["email"] != "user@example.com"
        
        # Should pass validation (no issues)
        assert len(result.issues) == 0
        
        # Should have redaction audit
        assert len(result.redaction_audit) == 1
        assert result.redaction_audit[0].path == "/email"
    
    @pytest.mark.asyncio
    async def test_batch_processing_integration(self):
        """Test batch processing with all components integrated."""
        # Create multiple page scenarios
        page_data = [
            {"page": "home", "user_id": "123", "valid": True},
            {"page": "about", "user_id": "456", "valid": True},
            {"page": "contact", "invalid_field": "value"},  # Missing user_id
            None  # Missing dataLayer
        ]
        
        contexts = [DLContext(url=f"https://example.com/page{i}") for i in range(len(page_data))]
        
        # Mock pages with different data
        mock_pages = []
        for i, data in enumerate(page_data):
            mock_page = AsyncMock()
            mock_page.url = contexts[i].url
            mock_page.evaluate.return_value = data
            mock_pages.append(mock_page)
        
        # Schema requiring user_id
        schema = {
            "type": "object",
            "properties": {
                "page": {"type": "string"},
                "user_id": {"type": "string"}
            },
            "required": ["user_id"]
        }
        
        # Process batch with aggregation
        aggregate = await self.service.batch_process_with_aggregation(
            pages=mock_pages,
            contexts=contexts,
            run_id="integration-test",
            schema=schema
        )
        
        # Verify aggregation results
        assert aggregate.run_id == "integration-test"
        assert aggregate.total_pages == 4
        assert aggregate.successful_captures == 3  # First 3 pages captured successfully
        assert aggregate.failed_captures == 1     # Last page failed
        
        # Should have variable statistics
        assert len(aggregate.variable_stats) > 0
        if "page" in aggregate.variable_stats:
            assert aggregate.variable_stats["page"]["presence"] > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])