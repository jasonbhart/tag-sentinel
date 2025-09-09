"""Seed list input provider for URL discovery from explicit sources.

This module handles URL input from seed lists, supporting both file input
and CLI argument lists with validation and error reporting.
"""

import asyncio
from typing import List, AsyncIterator, Optional, Union
from pathlib import Path
import logging
from urllib.parse import urlparse
import aiofiles

from ..models.crawl import PagePlan, DiscoveryMode
from ..utils.url_normalizer import is_valid_http_url, normalize


logger = logging.getLogger(__name__)


class SeedProviderError(Exception):
    """Raised when seed provider encounters an error."""
    pass


class SeedListProvider:
    """Provider for URLs from explicit seed lists.
    
    Supports multiple input formats:
    - List of URL strings (from CLI arguments)
    - Text files with one URL per line
    - UTF-8 encoding with comment support
    """
    
    def __init__(
        self,
        seeds: Optional[List[str]] = None,
        seed_files: Optional[List[Union[str, Path]]] = None,
        validate_urls: bool = True,
        skip_invalid: bool = True
    ):
        """Initialize seed list provider.
        
        Args:
            seeds: List of seed URL strings
            seed_files: List of file paths containing seed URLs
            validate_urls: Whether to validate URL format
            skip_invalid: Whether to skip invalid URLs or raise error
        """
        self.seeds = seeds or []
        self.seed_files = [Path(f) for f in (seed_files or [])]
        self.validate_urls = validate_urls
        self.skip_invalid = skip_invalid
        
        self._stats = {
            "total_discovered": 0,
            "valid_urls": 0,
            "invalid_urls": 0,
            "duplicate_urls": 0,
            "files_processed": 0,
            "files_failed": 0
        }
    
    async def discover_urls(self, depth: int = 0) -> AsyncIterator[PagePlan]:
        """Discover URLs from seed sources.
        
        Args:
            depth: Depth level for discovered URLs (typically 0 for seeds)
            
        Yields:
            PagePlan objects for each valid seed URL
        """
        seen_urls = set()
        
        # Process direct seed URLs
        for url in self.seeds:
            async for page_plan in self._process_url(url, depth, seen_urls, "direct"):
                yield page_plan
        
        # Process seed files
        for seed_file in self.seed_files:
            try:
                async for page_plan in self._process_file(seed_file, depth, seen_urls):
                    yield page_plan
                self._stats["files_processed"] += 1
            except Exception as e:
                self._stats["files_failed"] += 1
                error_msg = f"Failed to process seed file {seed_file}: {e}"
                logger.error(error_msg)
                if not self.skip_invalid:
                    raise SeedProviderError(error_msg)
        
        logger.info(f"Seed discovery completed: {self._stats}")
    
    async def _process_url(
        self,
        url: str,
        depth: int,
        seen_urls: set,
        source: str
    ) -> AsyncIterator[PagePlan]:
        """Process a single URL and create PagePlan if valid.
        
        Args:
            url: URL string to process
            depth: Depth level for the URL
            seen_urls: Set of already processed URLs for deduplication
            source: Source description for logging
            
        Yields:
            PagePlan if URL is valid and not duplicate
        """
        # Clean and normalize URL
        cleaned_url = url.strip()
        if not cleaned_url:
            return
        
        self._stats["total_discovered"] += 1
        
        try:
            # Validate URL format if requested
            if self.validate_urls:
                if not is_valid_http_url(cleaned_url):
                    self._stats["invalid_urls"] += 1
                    logger.debug(f"Invalid URL from {source}: {cleaned_url}")
                    if not self.skip_invalid:
                        raise SeedProviderError(f"Invalid URL: {cleaned_url}")
                    return
            
            # Normalize for deduplication
            normalized_url = normalize(cleaned_url)
            
            # Check for duplicates
            if normalized_url in seen_urls:
                self._stats["duplicate_urls"] += 1
                logger.debug(f"Duplicate URL from {source}: {normalized_url}")
                return
            
            seen_urls.add(normalized_url)
            self._stats["valid_urls"] += 1
            
            # Create PagePlan
            page_plan = PagePlan(
                url=normalized_url,
                source_url=None,  # Seeds have no source
                depth=depth,
                discovery_method="seeds",
                metadata={
                    "source": source,
                    "original_url": cleaned_url if cleaned_url != normalized_url else None
                }
            )
            
            logger.debug(f"Valid seed URL from {source}: {normalized_url}")
            yield page_plan
            
        except Exception as e:
            self._stats["invalid_urls"] += 1
            error_msg = f"Error processing URL '{cleaned_url}' from {source}: {e}"
            logger.warning(error_msg)
            if not self.skip_invalid:
                raise SeedProviderError(error_msg)
    
    async def _process_file(
        self,
        file_path: Path,
        depth: int,
        seen_urls: set
    ) -> AsyncIterator[PagePlan]:
        """Process URLs from a seed file.
        
        Args:
            file_path: Path to seed file
            depth: Depth level for discovered URLs
            seen_urls: Set of already processed URLs
            
        Yields:
            PagePlan objects for valid URLs in file
        """
        if not file_path.exists():
            raise SeedProviderError(f"Seed file does not exist: {file_path}")
        
        if not file_path.is_file():
            raise SeedProviderError(f"Seed path is not a file: {file_path}")
        
        logger.info(f"Processing seed file: {file_path}")
        
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                line_number = 0
                
                async for line in f:
                    line_number += 1
                    
                    # Skip empty lines and comments
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Handle multiple URLs per line (space or tab separated)
                    urls_in_line = line.split()
                    
                    for url in urls_in_line:
                        source = f"file:{file_path.name}:{line_number}"
                        async for page_plan in self._process_url(url, depth, seen_urls, source):
                            yield page_plan
                            
        except UnicodeDecodeError as e:
            raise SeedProviderError(f"Invalid UTF-8 encoding in file {file_path}: {e}")
        except Exception as e:
            raise SeedProviderError(f"Error reading seed file {file_path}: {e}")
    
    def get_stats(self) -> dict:
        """Get provider statistics.
        
        Returns:
            Dictionary with discovery statistics
        """
        return {
            "provider": "seed_list",
            **self._stats,
            "seed_count": len(self.seeds),
            "file_count": len(self.seed_files),
            "validation_enabled": self.validate_urls,
            "skip_invalid": self.skip_invalid
        }
    
    @classmethod
    async def from_sources(
        cls,
        seeds: Optional[List[str]] = None,
        seed_files: Optional[List[Union[str, Path]]] = None,
        **kwargs
    ) -> List[PagePlan]:
        """Convenience method to get all PagePlans from sources.
        
        Args:
            seeds: List of seed URLs
            seed_files: List of seed file paths
            **kwargs: Additional arguments for SeedListProvider
            
        Returns:
            List of all discovered PagePlans
        """
        provider = cls(seeds=seeds, seed_files=seed_files, **kwargs)
        page_plans = []
        
        async for page_plan in provider.discover_urls():
            page_plans.append(page_plan)
        
        return page_plans
    
    @classmethod
    def validate_seed_sources(
        cls,
        seeds: Optional[List[str]] = None,
        seed_files: Optional[List[Union[str, Path]]] = None
    ) -> List[str]:
        """Validate seed sources without processing URLs.
        
        Args:
            seeds: List of seed URLs to validate
            seed_files: List of seed file paths to validate
            
        Returns:
            List of validation errors (empty if all valid)
        """
        errors = []
        
        # Validate seed URLs format
        if seeds:
            for i, seed in enumerate(seeds):
                if not isinstance(seed, str) or not seed.strip():
                    errors.append(f"Seed {i+1}: Empty or invalid URL")
                    continue
                
                try:
                    parsed = urlparse(seed.strip())
                    if not parsed.scheme or not parsed.netloc:
                        errors.append(f"Seed {i+1}: Invalid URL format: {seed}")
                except Exception:
                    errors.append(f"Seed {i+1}: Cannot parse URL: {seed}")
        
        # Validate seed files
        if seed_files:
            for seed_file in seed_files:
                file_path = Path(seed_file)
                if not file_path.exists():
                    errors.append(f"Seed file does not exist: {seed_file}")
                elif not file_path.is_file():
                    errors.append(f"Seed path is not a file: {seed_file}")
                elif not file_path.stat().st_size:
                    errors.append(f"Seed file is empty: {seed_file}")
        
        return errors