# Tag Sentinel Tests

This directory contains the test suite for Tag Sentinel's EPIC 1 implementation.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures and configuration
├── README.md               # This file
├── unit/                   # Unit tests for individual components
│   ├── __init__.py
│   ├── test_url_normalizer.py
│   ├── test_scope_matcher.py
│   ├── test_models.py
│   └── test_frontier_queue.py
├── integration/            # Integration tests for complete workflows
│   ├── __init__.py
│   └── test_epic1_integration.py
└── fixtures/               # Test data and utilities
    └── __init__.py
```

## Running Tests

### Prerequisites
Ensure you have the virtual environment activated and dependencies installed:

```bash
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
```

### Run All Tests
```bash
# Using pytest
python -m pytest tests/ -v

# Using our test runner
python run_tests.py
```

### Run Specific Test Categories

**Unit Tests Only:**
```bash
python -m pytest tests/unit/ -v
# or
python run_tests.py unit
```

**Integration Tests Only:**
```bash
python -m pytest tests/integration/ -v  
# or
python run_tests.py integration
```

**Quick Tests (exclude slow ones):**
```bash
python run_tests.py quick
```

### Run Tests with Coverage
```bash
python run_tests.py coverage
```

### Run Specific Test Files
```bash
python -m pytest tests/unit/test_url_normalizer.py -v
python -m pytest tests/integration/test_epic1_integration.py -v
```

## Test Configuration

Tests are configured via `pytest.ini` in the project root:
- Async tests are automatically handled
- Warnings are filtered for cleaner output
- Test markers are defined for categorization

## Writing New Tests

### Unit Tests
Create unit tests in `tests/unit/` following the pattern `test_<component>.py`:

```python
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.audit.utils.example import ExampleClass

class TestExampleClass:
    def test_basic_functionality(self):
        instance = ExampleClass()
        assert instance.method() == expected_value
```

### Integration Tests
Integration tests should test complete workflows and component interactions.

### Async Tests
Use `@pytest.mark.asyncio` for async test functions:

```python
@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_function()
    assert result is not None
```

## Test Coverage

Current test coverage focuses on EPIC 1 components:
- ✅ URL Normalization (`app.audit.utils.url_normalizer`)
- ✅ Scope Matching (`app.audit.utils.scope_matcher`)
- ✅ Data Models (`app.audit.models.crawl`)
- ✅ Frontier Queue (`app.audit.queue.frontier_queue`)
- ✅ Rate Limiting (`app.audit.queue.rate_limiter`)
- ✅ Input Providers (`app.audit.input.*`)
- ✅ Main Crawler (`app.audit.crawler`)

## Fixtures

Shared test fixtures are defined in `conftest.py`:
- `sample_crawl_config`: Basic crawl configuration
- `sample_page_plan`: Sample page plan object
- `frontier_queue`: Frontier queue instance
- `temp_seed_file`: Temporary seed file for file-based tests

## CI/CD Integration

Tests are designed to run in CI environments. The test runner script provides different modes:
- `all`: Complete test suite
- `unit`: Unit tests only (fast)
- `integration`: Integration tests
- `quick`: Exclude slow tests
- `coverage`: Generate coverage reports