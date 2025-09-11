# Tag Sentinel

[![Tests](https://img.shields.io/badge/tests-35%2F35%20passing-green)](./tests/)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)

**Open-source web analytics auditing and monitoring platform** similar to commercial solutions (OP), built with Playwright for browser automation and Python for the backend.

Tag Sentinel audits and monitors website analytics instrumentation, tag implementations, and data layer integrity to ensure your analytics tracking works correctly across all environments.

## 🎯 Core Capabilities

- **📊 Analytics Tracking Validation**: Verify GA4, GTM, and other analytics tags are firing correctly
- **🔄 Tag Load Sequencing**: Ensure tags fire in the correct order when dependencies exist
- **🍪 Cookie Usage Analysis**: Inventory cookies set by analytics/marketing tags for privacy compliance
- **📋 Data Layer Integrity**: Validate structure and content of website data layers (window.dataLayer)
- **⚡ Tag Performance Monitoring**: Monitor load times and error status of tag requests
- **🔍 Duplicate/Missing Tag Detection**: Identify duplicate tag firings or missing expected tags
- **🌐 Multi-Environment Support**: Run audits across development, staging, and production environments

## 🏗️ Architecture

Tag Sentinel is built with a modular architecture designed for scalability and extensibility:

- **🕷️ Playwright Crawling Engine**: Headless browser automation for realistic page loading and tag capture
- **🐍 Python Backend**: Core application handling audit orchestration, data processing, and API endpoints
- **🎛️ Web Interface**: Dashboard for viewing audit results, managing configurations, and monitoring
- **💻 CLI Tool**: Command-line interface for local debugging and testing
- **🔧 Modular Tag System**: Extensible framework for supporting different analytics platforms (starting with GA4/GTM)

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js (for Playwright browsers)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/tag-sentinel.git
   cd tag-sentinel
   ```

2. **Set up Python environment**

   ```bash
   # Create virtual environment
   python -m venv .venv

   # Activate virtual environment
   source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Install Playwright browsers** (for browser capture)

   ```bash
   python -m playwright install
   ```

### Running Tests

```bash
# Run all tests (unit + integration)
pytest tests/ -v

# Run only unit tests
pytest tests/unit/ -v

# Run only integration tests
pytest tests/integration/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

## 🚀 Getting Started

The quickest way to try Tag Sentinel is to capture a page in a real browser and print a short summary.

```python
# save as quick_start.py
import asyncio
from app.audit.capture.engine import create_capture_engine

async def main():
    engine = create_capture_engine(headless=True)
    async with engine.session():
        result = await engine.capture_page("https://example.com")
        print({
            "url": result.url,
            "status": str(result.capture_status),
            "requests": len(result.network_requests or []),
            "console": len(result.console_logs or []),
        })

asyncio.run(main())
```

Run it:

```bash
python quick_start.py
```

## 🔧 Usage Examples

- Capture a page and run detectors (GA4/GTM):

```python
import asyncio
from app.audit.capture.engine import create_capture_engine
from app.audit.detectors import GA4Detector, GTMDetector, DetectContext

async def main():
    engine = create_capture_engine(headless=True)
    async with engine.session():
        page_result = await engine.capture_page("https://example.com")

        ctx = DetectContext(environment="development", is_production=False)
        ga4_result = await GA4Detector().detect(page_result, ctx)
        gtm_result = GTMDetector().detect(page_result, ctx)

        print("GA4 events:", len(ga4_result.events))
        print("GTM events:", len(gtm_result.events))

asyncio.run(main())
```

- Capture and validate the page dataLayer with schema support:

```python
import asyncio
from app.audit.capture.browser_factory import create_default_factory
from app.audit.datalayer.service import DataLayerService

async def main():
    factory = create_default_factory()
    await factory.start()
    try:
        async with factory.page() as page:
            await page.goto("https://example.com", timeout=30000)

            service = DataLayerService()  # configure schema via service.config if needed
            dl_result = await service.capture_and_validate(page, "https://example.com")

            print("dataLayer found:", dl_result.snapshot.exists)
            print("validation issues:", len(dl_result.issues))
    finally:
        await factory.stop()

asyncio.run(main())
```

## 🛠️ Development

### Project Structure

```
tag-sentinel/
├── app/
│   └── audit/
│       ├── models/          # Pydantic data models
│       ├── utils/           # URL normalization, scope matching
│       ├── queue/           # Frontier queue, rate limiting
│       ├── input/           # URL discovery providers
│       └── crawler.py       # Main crawling orchestrator
├── tests/
│   ├── unit/               # Component unit tests
│   └── integration/        # End-to-end integration tests
├── requirements.txt        # Python dependencies
└── pytest.ini            # Test configuration
```

### Key Dependencies

- **[Pydantic](https://docs.pydantic.dev/) 2.11.7+**: Data validation and serialization
- **[aiohttp](https://docs.aiohttp.org/) 3.12.15+**: Async HTTP client for sitemap fetching
- **[aiofiles](https://github.com/Tinche/aiofiles) 24.1.0+**: Async file operations
- **[pytest](https://docs.pytest.org/) 8.4.0+**: Testing framework with async support

### Running the Development Environment

```bash
# Activate virtual environment
source .venv/bin/activate

# Run tests during development
pytest tests/ -v --tb=short

# Run specific test file
pytest tests/unit/test_crawler.py -v

# Test with verbose output
pytest tests/integration/ -v -s
```

### Code Quality

```bash
# Type checking (when implemented)
mypy app/
```

## 🎯 Design Principles

- **🌐 Browser-First Approach**: Uses real browser rendering via Playwright to capture exactly what users experience
- **🔧 Modular & Extensible**: Plugin-style architecture for adding new analytics platforms
- **🏢 Multi-Environment**: Supports testing across dev/staging/prod with different configurations
- **📋 Rule-Based Validation**: Configurable rules for validating tag presence, parameters, and data layer values
- **⚡ Performance Conscious**: Concurrent page processing with configurable rate limiting
- **👨‍💻 Developer-Friendly**: Designed for integration into CI/CD pipelines and development workflows

## 📋 Configuration

Tag Sentinel uses YAML configuration files for:

- **Site crawl settings**: URLs, authentication, concurrency limits
- **Tag detection rules**: Validation criteria for different analytics platforms
- **Environment-specific settings**: Different tracking IDs for staging vs production
- **Scheduling configuration**: For periodic automated audits
- **Output and reporting preferences**: Format and destination of audit results

Example configuration structure:

```yaml
sites:
  production:
    crawl:
      max_pages: 1000
      max_concurrency: 5
      include: [".*\\.example\\.com.*"]
      exclude: [".*/admin/.*"]
    tags:
      ga4_measurement_id: "G-XXXXXXXXXX"
      gtm_container_id: "GTM-XXXXXXX"
```

- Detector configuration lives in `config/detectors.yaml` and controls GA4/GTM detection, MP debug, duplicate windows, and sequencing rules.
- Data Layer validation can be enabled programmatically via `DataLayerService` (schema paths in JSON or YAML files are supported).

## 🤝 Contributing

We welcome contributions! Tag Sentinel is designed to be:

1. **Extensible**: Easy to add new analytics platforms and tag types
2. **Testable**: Comprehensive test coverage with both unit and integration tests
3. **Configurable**: YAML-based configuration for different environments and use cases
4. **Performant**: Async architecture with proper rate limiting and concurrency controls

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Implement the feature
5. Ensure all tests pass
6. Submit a pull request

## 📚 Documentation

- [Implementation Guide](./implementation.md) - Detailed technical implementation and architecture
- [Development Guide](./CLAUDE.md) - Development environment and workflow guidance
- [Test Documentation](./tests/README.md) - Testing strategy and test organization

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## 🙏 Acknowledgments

Inspired by enterprise solutions like ObservePoint and built for the open-source community to democratize web analytics auditing and monitoring.

[![Built with Python](https://img.shields.io/badge/Built%20with-Python-3776ab.svg)](https://www.python.org/)
[![Powered by Playwright](https://img.shields.io/badge/Powered%20by-Playwright-45ba4b.svg)](https://playwright.dev/)
[![Async Architecture](https://img.shields.io/badge/Architecture-Async-brightgreen.svg)](https://docs.python.org/3/library/asyncio.html)
