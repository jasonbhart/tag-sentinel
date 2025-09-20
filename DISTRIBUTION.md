# Package Distribution Guide

This guide covers how to build, test, and distribute Tag Sentinel packages.

## Prerequisites

- Python 3.9 or higher
- uv (recommended) or pip for package management
- setuptools and wheel for building

## Building Packages

### Using uv (Recommended)

```bash
# Build both wheel and source distribution
uv build

# Build only wheel
uv build --wheel

# Build only source distribution
uv build --sdist
```

### Using traditional tools

```bash
# Install build dependencies
pip install build

# Build packages
python -m build
```

## Package Contents

The built packages include:

- **Source Distribution** (`tag_sentinel-1.0.0.tar.gz`):
  - Complete source code
  - All configuration files
  - Documentation
  - License and metadata

- **Wheel Distribution** (`tag_sentinel-1.0.0-py3-none-any.whl`):
  - Compiled Python code
  - Ready for installation
  - Platform-independent

## Console Scripts

The package provides two console scripts:

- `openaudit`: Primary CLI entry point
- `tag-sentinel`: Alternative CLI entry point

Both scripts point to the same Typer-based CLI implementation.

## Testing the Package

### Local Installation

```bash
# Install from wheel
pip install dist/tag_sentinel-1.0.0-py3-none-any.whl

# Install from source
pip install dist/tag_sentinel-1.0.0.tar.gz

# Install in development mode
pip install -e .
```

### Verify Installation

```bash
# Test CLI availability
openaudit --help
tag-sentinel --help

# Test version command
openaudit version

# Test basic functionality
openaudit run --help
```

### Test in Clean Environment

```bash
# Create test environment
python -m venv test_env
source test_env/bin/activate  # Windows: test_env\Scripts\activate

# Install package
pip install dist/tag_sentinel-1.0.0-py3-none-any.whl

# Test functionality
openaudit run https://httpbin.org/json

# Clean up
deactivate
rm -rf test_env
```

## Publishing to PyPI

### Test PyPI (Recommended First)

```bash
# Install twine
pip install twine

# Upload to test PyPI
twine upload --repository testpypi dist/*

# Test installation from test PyPI
pip install --index-url https://test.pypi.org/simple/ tag-sentinel
```

### Production PyPI

```bash
# Upload to production PyPI
twine upload dist/*

# Test installation
pip install tag-sentinel
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build and Publish

on:
  release:
    types: [published]

jobs:
  build-and-publish:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine

    - name: Build package
      run: python -m build

    - name: Check package
      run: twine check dist/*

    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

## Version Management

The version is managed in `pyproject.toml`:

```toml
[project]
version = "1.0.0"
```

For releases:

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create git tag: `git tag v1.0.0`
4. Push tag: `git push origin v1.0.0`
5. Create GitHub release
6. CI/CD will automatically build and publish

## Quality Checks

Before publishing, run these checks:

```bash
# Type checking
mypy app/

# Code formatting
black --check app/

# Linting
flake8 app/

# Test suite
pytest tests/

# Package validation
twine check dist/*

# Dependency checking
pip-audit  # If available
```

## Troubleshooting

### Common Issues

1. **ImportError after installation**:
   - Check that all dependencies are properly specified
   - Verify package structure in `pyproject.toml`

2. **Console scripts not working**:
   - Verify `[project.scripts]` configuration
   - Check entry point function exists and is importable

3. **Missing files in distribution**:
   - Check `[tool.setuptools.packages.find]` configuration
   - Use `MANIFEST.in` for additional files if needed

4. **Dependency conflicts**:
   - Review version constraints in `dependencies`
   - Test in clean environments

### Package Inspection

```bash
# List wheel contents
unzip -l dist/tag_sentinel-1.0.0-py3-none-any.whl

# Extract and inspect
tar -tzf dist/tag_sentinel-1.0.0.tar.gz

# Check metadata
pip show tag-sentinel
```

## Security Considerations

- Use API tokens for PyPI authentication
- Never commit credentials to version control
- Use environment variables or CI/CD secrets
- Consider signing packages with GPG
- Regularly update dependencies for security patches

## Support

For issues with package distribution:

1. Check this guide first
2. Review build logs for errors
3. Test in clean environments
4. Open issue on GitHub repository