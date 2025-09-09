"""Shared test fixtures and configuration for Tag Sentinel tests."""

import pytest
import asyncio
import tempfile
from pathlib import Path
import sys

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.audit.models.crawl import CrawlConfig, DiscoveryMode, PagePlan
from app.audit.queue.frontier_queue import FrontierQueue


@pytest.fixture
def sample_crawl_config():
    """Sample crawl configuration for testing."""
    return CrawlConfig(
        discovery_mode=DiscoveryMode.SEEDS,
        seeds=["https://example.com", "https://example.com/about"],
        max_pages=10,
        max_concurrency=2,
        include_patterns=[".*example\\.com.*"],
        exclude_patterns=[".*/admin/.*"],
        same_site_only=True
    )


@pytest.fixture
def sample_page_plan():
    """Sample page plan for testing."""
    return PagePlan(
        url="https://example.com/page",
        depth=0,
        discovery_method="seeds",
        metadata={"source": "test"}
    )


@pytest.fixture
def sample_page_plans():
    """List of sample page plans for testing."""
    return [
        PagePlan(url=f"https://example.com/page{i}", depth=i, discovery_method="seeds")
        for i in range(5)
    ]


@pytest.fixture
async def frontier_queue():
    """Frontier queue instance for testing."""
    queue = FrontierQueue(max_size=100)
    yield queue
    await queue.close()


@pytest.fixture
def temp_seed_file():
    """Temporary seed file for testing."""
    content = """
# Test seed file
https://example.com
https://example.com/page1
https://example.com/page2

# This is a comment
https://example.com/page3
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content.strip())
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Test configuration
pytest_plugins = []


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )