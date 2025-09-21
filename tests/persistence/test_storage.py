"""Tests for artifact storage backends."""

import json
import pytest
from pathlib import Path
from typing import Dict, Any

from app.persistence.storage import (
    LocalArtifactStore,
    ArtifactStore,
    create_artifact_store,
    create_default_artifact_store
)


class TestLocalArtifactStore:
    """Test local filesystem artifact storage."""

    async def test_store_and_retrieve_bytes(self, artifact_store: LocalArtifactStore):
        """Test storing and retrieving byte content."""
        content = b"Hello, world!"
        path = "test/hello.txt"

        # Store content
        artifact_ref = await artifact_store.put(content, path, content_type="text/plain")

        assert artifact_ref.path == path
        assert artifact_ref.size_bytes == len(content)
        assert artifact_ref.content_type == "text/plain"
        assert len(artifact_ref.checksum) == 64  # SHA-256 hex

        # Check if artifact exists
        exists = await artifact_store.exists(path)
        assert exists is True

        # Get URL
        url = await artifact_store.get_url(path)
        assert url.startswith("file://")
        assert path in url

    async def test_store_and_retrieve_string(self, artifact_store: LocalArtifactStore):
        """Test storing string content."""
        content = "Hello, world!"
        path = "test/hello_string.txt"

        artifact_ref = await artifact_store.put(content, path)

        assert artifact_ref.path == path
        assert artifact_ref.size_bytes == len(content.encode())

    async def test_store_with_metadata(self, artifact_store: LocalArtifactStore):
        """Test storing content with metadata."""
        content = b'{"key": "value"}'
        path = "test/metadata.json"
        metadata = {"source": "test", "version": 1}

        artifact_ref = await artifact_store.put(
            content,
            path,
            content_type="application/json",
            metadata=metadata
        )

        # Verify metadata is stored
        retrieved_metadata = await artifact_store.get_metadata(path)
        assert retrieved_metadata is not None
        assert retrieved_metadata["source"] == "test"
        assert retrieved_metadata["version"] == 1
        assert retrieved_metadata["content_type"] == "application/json"

    async def test_delete_artifact(self, artifact_store: LocalArtifactStore):
        """Test deleting an artifact."""
        content = b"to be deleted"
        path = "test/delete_me.txt"

        # Store then delete
        await artifact_store.put(content, path)

        # Verify it exists
        exists_before = await artifact_store.exists(path)
        assert exists_before is True

        # Delete it
        deleted = await artifact_store.delete(path)
        assert deleted is True

        # Verify it's gone
        exists_after = await artifact_store.exists(path)
        assert exists_after is False

        # Deleting again should return False
        deleted_again = await artifact_store.delete(path)
        assert deleted_again is False

    async def test_get_url_nonexistent(self, artifact_store: LocalArtifactStore):
        """Test getting URL for non-existent artifact."""
        with pytest.raises(FileNotFoundError):
            await artifact_store.get_url("nonexistent/file.txt")

    async def test_get_metadata_nonexistent(self, artifact_store: LocalArtifactStore):
        """Test getting metadata for non-existent artifact."""
        metadata = await artifact_store.get_metadata("nonexistent/file.txt")
        assert metadata is None

    async def test_nested_paths(self, artifact_store: LocalArtifactStore):
        """Test storing artifacts in nested directory structure."""
        content = b"nested content"
        path = "deep/nested/directory/file.txt"

        artifact_ref = await artifact_store.put(content, path)

        assert artifact_ref.path == path

        exists = await artifact_store.exists(path)
        assert exists is True

    async def test_store_file_path(self, artifact_store: LocalArtifactStore, temp_artifact_dir: Path):
        """Test storing content from a file path."""
        # Create a temporary file
        temp_file = temp_artifact_dir / "source.txt"
        temp_file.write_text("File content")

        path = "test/from_file.txt"
        artifact_ref = await artifact_store.put(temp_file, path)

        assert artifact_ref.path == path
        assert artifact_ref.size_bytes == len("File content".encode())

        # Verify content matches
        url = await artifact_store.get_url(path)
        stored_file = Path(url.replace("file://", ""))
        assert stored_file.read_text() == "File content"


class TestStorageFactory:
    """Test storage factory functions."""

    def test_create_local_artifact_store(self, temp_artifact_dir: Path):
        """Test creating local artifact store via factory."""
        store = create_artifact_store("local", base_path=temp_artifact_dir)

        assert isinstance(store, LocalArtifactStore)
        assert store.base_path == temp_artifact_dir

    def test_create_unsupported_backend(self):
        """Test creating unsupported storage backend."""
        with pytest.raises(ValueError, match="Unsupported storage backend"):
            create_artifact_store("unsupported")

    def test_create_default_local_store(self, monkeypatch, temp_artifact_dir: Path):
        """Test creating default local store from environment."""
        monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
        monkeypatch.setenv("ARTIFACT_STORAGE_PATH", str(temp_artifact_dir))

        store = create_default_artifact_store()

        assert isinstance(store, LocalArtifactStore)
        assert store.base_path == temp_artifact_dir

    def test_create_default_unsupported_backend(self, monkeypatch):
        """Test creating default store with unsupported backend."""
        monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "unsupported")

        with pytest.raises(ValueError, match="Unsupported storage backend"):
            create_default_artifact_store()


class TestChecksumCalculation:
    """Test checksum calculation."""

    async def test_consistent_checksums(self, artifact_store: LocalArtifactStore):
        """Test that same content produces same checksum."""
        content = b"consistent content"
        path1 = "test/checksum1.txt"
        path2 = "test/checksum2.txt"

        ref1 = await artifact_store.put(content, path1)
        ref2 = await artifact_store.put(content, path2)

        assert ref1.checksum == ref2.checksum

    async def test_different_checksums(self, artifact_store: LocalArtifactStore):
        """Test that different content produces different checksums."""
        content1 = b"content one"
        content2 = b"content two"
        path1 = "test/diff1.txt"
        path2 = "test/diff2.txt"

        ref1 = await artifact_store.put(content1, path1)
        ref2 = await artifact_store.put(content2, path2)

        assert ref1.checksum != ref2.checksum


class TestArtifactRef:
    """Test ArtifactRef functionality."""

    async def test_artifact_ref_properties(self, artifact_store: LocalArtifactStore):
        """Test ArtifactRef contains expected properties."""
        content = b'{"test": true}'
        path = "test/ref_test.json"
        metadata = {"created_by": "test"}

        ref = await artifact_store.put(
            content,
            path,
            content_type="application/json",
            metadata=metadata
        )

        assert ref.path == path
        assert ref.size_bytes == len(content)
        assert ref.content_type == "application/json"
        assert ref.metadata == metadata
        assert isinstance(ref.checksum, str)
        assert len(ref.checksum) == 64  # SHA-256


# Note: S3 tests would require either:
# 1. Mocking boto3/aiobotocore (complex due to async context managers)
# 2. Integration tests with real S3/LocalStack (beyond unit test scope)
# 3. Separate S3-specific test file with proper setup
#
# For now, we focus on the local storage implementation and factory functions.
# S3 functionality can be tested in integration tests or with proper mocking setup.