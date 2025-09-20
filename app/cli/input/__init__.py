"""CLI input loaders for different URL discovery methods."""

from .sitemap_loader import load_urls_from_sitemap
from .crawl_loader import load_urls_from_crawl

__all__ = [
    "load_urls_from_sitemap",
    "load_urls_from_crawl"
]