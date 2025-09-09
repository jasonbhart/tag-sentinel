"""Unit tests for crawl data models."""

import pytest
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.audit.models.crawl import (
    CrawlConfig,
    DiscoveryMode,
    LoadWaitStrategy,
    PagePlan,
    CrawlStats,
    CrawlMetrics
)


class TestCrawlConfig:
    """Test cases for CrawlConfig model."""
    
    def test_basic_config_creation(self):
        """Test basic configuration creation."""
        config = CrawlConfig(
            discovery_mode=DiscoveryMode.SEEDS,
            seeds=["https://example.com"],
            max_pages=100,
            include_patterns=[".*example.*"]
        )
        
        assert config.discovery_mode == DiscoveryMode.SEEDS
        assert len(config.seeds) == 1
        assert config.max_pages == 100
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Should require sitemap_url for sitemap mode
        with pytest.raises(ValueError):
            CrawlConfig(
                discovery_mode=DiscoveryMode.SITEMAP,
                max_pages=100
            )
        
        # Should require seeds for seed mode
        with pytest.raises(ValueError):
            CrawlConfig(
                discovery_mode=DiscoveryMode.SEEDS,
                max_pages=100
            )
    
    def test_regex_pattern_validation(self):
        """Test regex pattern validation."""
        with pytest.raises(ValueError):
            CrawlConfig(
                discovery_mode=DiscoveryMode.SEEDS,
                seeds=["https://example.com"],
                include_patterns=["[invalid regex"]
            )


class TestPagePlan:
    """Test cases for PagePlan model."""
    
    def test_basic_pageplan_creation(self):
        """Test basic PagePlan creation."""
        page_plan = PagePlan(
            url="https://example.com",
            depth=0,
            discovery_method="seeds"
        )
        
        assert str(page_plan.url) == "https://example.com/"  # Pydantic adds trailing slash
        assert page_plan.depth == 0
        assert page_plan.discovery_method == "seeds"
        assert isinstance(page_plan.discovered_at, datetime)
    
    def test_pageplan_with_metadata(self):
        """Test PagePlan with metadata."""
        metadata = {"source": "test", "priority": "high"}
        page_plan = PagePlan(
            url="https://example.com/page",
            source_url="https://example.com",
            depth=1,
            discovery_method="dom",
            metadata=metadata
        )
        
        assert page_plan.metadata == metadata
        assert str(page_plan.source_url) == "https://example.com/"


class TestCrawlStats:
    """Test cases for CrawlStats model."""
    
    def test_stats_initialization(self):
        """Test stats initialization with defaults."""
        stats = CrawlStats()
        
        assert stats.urls_discovered == 0
        assert stats.urls_processed == 0
        assert stats.success_rate == 0.0
        assert stats.duration is None
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        stats = CrawlStats()
        stats.urls_processed = 80
        stats.urls_failed = 20
        
        assert stats.success_rate == 80.0
    
    def test_duration_calculation(self):
        """Test duration calculation."""
        stats = CrawlStats()
        start = datetime.utcnow()
        stats.start_time = start
        stats.end_time = datetime.utcnow()
        
        duration = stats.duration
        assert duration is not None
        assert duration >= 0


class TestCrawlMetrics:
    """Test cases for CrawlMetrics model."""
    
    def test_metrics_creation(self):
        """Test metrics creation with config."""
        config = CrawlConfig(
            discovery_mode=DiscoveryMode.SEEDS,
            seeds=["https://example.com"],
            max_pages=100
        )
        
        metrics = CrawlMetrics(config=config)
        
        assert metrics.config == config
        assert isinstance(metrics.stats, CrawlStats)
        assert metrics.is_running is False
    
    def test_error_tracking(self):
        """Test error tracking functionality."""
        config = CrawlConfig(
            discovery_mode=DiscoveryMode.SEEDS,
            seeds=["https://example.com"],
            max_pages=100
        )
        
        metrics = CrawlMetrics(config=config)
        metrics.add_error("https://example.com/error", "timeout", "Page timeout")
        
        assert len(metrics.recent_errors) == 1
        error = metrics.recent_errors[0]
        assert error["url"] == "https://example.com/error"
        assert error["error_type"] == "timeout"
    
    def test_export_summary(self):
        """Test metrics export summary."""
        config = CrawlConfig(
            discovery_mode=DiscoveryMode.SEEDS,
            seeds=["https://example.com"],
            max_pages=100
        )
        
        metrics = CrawlMetrics(config=config)
        summary = metrics.export_summary()
        
        assert "discovery_mode" in summary
        assert "urls_processed" in summary
        assert "success_rate" in summary


if __name__ == "__main__":
    pytest.main([__file__])