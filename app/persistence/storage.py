"""Artifact storage backends for Tag Sentinel persistence layer.

This module provides abstract and concrete implementations for storing audit artifacts
in different backends (local filesystem, S3-compatible object storage).
"""

import hashlib
import os
import shutil
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union, BinaryIO, Dict, Any
from urllib.parse import urlparse

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

try:
    import aiobotocore.session
    from botocore.exceptions import ClientError as AsyncClientError
    HAS_AIOBOTOCORE = True
except ImportError:
    HAS_AIOBOTOCORE = False


class ArtifactRef:
    """Reference to a stored artifact with metadata."""

    def __init__(
        self,
        path: str,
        checksum: str,
        size_bytes: int,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.path = path
        self.checksum = checksum
        self.size_bytes = size_bytes
        self.content_type = content_type
        self.metadata = metadata or {}


class ArtifactStore(ABC):
    """Abstract base class for artifact storage backends."""

    @abstractmethod
    async def put(
        self,
        content: Union[bytes, BinaryIO, str, Path],
        path: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ArtifactRef:
        """Store artifact content and return reference.

        Args:
            content: Content to store (bytes, file-like object, or file path)
            path: Storage path/key for the artifact
            content_type: MIME content type
            metadata: Additional metadata to store with artifact

        Returns:
            ArtifactRef with storage details
        """
        pass

    @abstractmethod
    async def get_url(
        self,
        path: str,
        signed: bool = True,
        ttl_seconds: int = 3600
    ) -> str:
        """Get URL to access artifact.

        Args:
            path: Storage path/key of the artifact
            signed: Whether to generate signed URL (for restricted access)
            ttl_seconds: Time-to-live for signed URLs

        Returns:
            URL to access the artifact
        """
        pass

    @abstractmethod
    async def delete(self, path: str) -> bool:
        """Delete artifact from storage.

        Args:
            path: Storage path/key of the artifact

        Returns:
            True if deleted successfully, False if not found
        """
        pass

    @abstractmethod
    async def exists(self, path: str) -> bool:
        """Check if artifact exists in storage.

        Args:
            path: Storage path/key of the artifact

        Returns:
            True if artifact exists
        """
        pass

    @abstractmethod
    async def get_metadata(self, path: str) -> Optional[Dict[str, Any]]:
        """Get artifact metadata.

        Args:
            path: Storage path/key of the artifact

        Returns:
            Metadata dictionary or None if not found
        """
        pass

    def _calculate_checksum(self, content: bytes) -> str:
        """Calculate SHA-256 checksum of content."""
        return hashlib.sha256(content).hexdigest()

    def _read_content(self, content: Union[bytes, BinaryIO, str, Path]) -> bytes:
        """Read content from various input types."""
        if isinstance(content, bytes):
            return content
        elif isinstance(content, (str, Path)):
            with open(content, 'rb') as f:
                return f.read()
        else:
            # Assume file-like object
            if hasattr(content, 'read'):
                return content.read()
            else:
                raise ValueError(f"Unsupported content type: {type(content)}")


class LocalArtifactStore(ArtifactStore):
    """Local filesystem artifact storage backend."""

    def __init__(self, base_path: Union[str, Path] = "./artifacts"):
        """Initialize local storage.

        Args:
            base_path: Base directory for artifact storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def put(
        self,
        content: Union[bytes, BinaryIO, str, Path],
        path: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ArtifactRef:
        """Store artifact in local filesystem."""
        # Read content and calculate checksum
        content_bytes = self._read_content(content)
        checksum = self._calculate_checksum(content_bytes)
        size_bytes = len(content_bytes)

        # Create full file path
        file_path = self.base_path / path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write content to file
        with open(file_path, 'wb') as f:
            f.write(content_bytes)

        # Store metadata in sidecar file if provided
        if metadata:
            metadata_path = file_path.with_suffix(file_path.suffix + '.meta')
            import json
            with open(metadata_path, 'w') as f:
                json.dump({
                    'content_type': content_type,
                    'checksum': checksum,
                    'size_bytes': size_bytes,
                    'created_at': datetime.utcnow().isoformat(),
                    **metadata
                }, f, indent=2)

        return ArtifactRef(
            path=path,
            checksum=checksum,
            size_bytes=size_bytes,
            content_type=content_type,
            metadata=metadata
        )

    async def get_url(
        self,
        path: str,
        signed: bool = True,
        ttl_seconds: int = 3600
    ) -> str:
        """Get file:// URL for local access."""
        file_path = self.base_path / path
        if not file_path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")

        # For local storage, return file:// URL
        return file_path.absolute().as_uri()

    async def delete(self, path: str) -> bool:
        """Delete artifact and metadata files."""
        file_path = self.base_path / path
        metadata_path = file_path.with_suffix(file_path.suffix + '.meta')

        deleted = False
        if file_path.exists():
            file_path.unlink()
            deleted = True

        if metadata_path.exists():
            metadata_path.unlink()

        return deleted

    async def exists(self, path: str) -> bool:
        """Check if artifact file exists."""
        file_path = self.base_path / path
        return file_path.exists()

    async def get_metadata(self, path: str) -> Optional[Dict[str, Any]]:
        """Get metadata from sidecar file."""
        file_path = self.base_path / path
        metadata_path = file_path.with_suffix(file_path.suffix + '.meta')

        if not metadata_path.exists():
            return None

        import json
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None


class S3ArtifactStore(ArtifactStore):
    """S3-compatible object storage backend with async support."""

    # 5GB threshold for multipart uploads (AWS S3 limit for single PUT)
    MULTIPART_THRESHOLD = 5 * 1024 * 1024 * 1024  # 5GB
    # 100MB chunk size for multipart uploads
    MULTIPART_CHUNK_SIZE = 100 * 1024 * 1024  # 100MB

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region: str = "us-east-1",
        endpoint_url: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None
    ):
        """Initialize S3 storage.

        Args:
            bucket: S3 bucket name
            prefix: Key prefix for all artifacts
            region: AWS region
            endpoint_url: Custom endpoint URL (for S3-compatible services)
            aws_access_key_id: AWS access key (optional, can use IAM)
            aws_secret_access_key: AWS secret key (optional, can use IAM)
        """
        if not HAS_AIOBOTOCORE:
            raise ImportError("aiobotocore is required for async S3 storage backend")

        self.bucket = bucket
        self.prefix = prefix.rstrip('/') + '/' if prefix else ''
        self.region = region

        # Store configuration for creating async sessions
        self._config = {
            'region_name': region
        }
        if endpoint_url:
            self._config['endpoint_url'] = endpoint_url
        if aws_access_key_id and aws_secret_access_key:
            self._config['aws_access_key_id'] = aws_access_key_id
            self._config['aws_secret_access_key'] = aws_secret_access_key

        # Create aiobotocore session
        self._session = aiobotocore.session.AioSession()

    def _get_key(self, path: str) -> str:
        """Get full S3 key with prefix."""
        return self.prefix + path.lstrip('/')

    async def put(
        self,
        content: Union[bytes, BinaryIO, str, Path],
        path: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ArtifactRef:
        """Store artifact in S3 with multipart upload support for large files."""
        # Read content and calculate checksum
        content_bytes = self._read_content(content)
        checksum = self._calculate_checksum(content_bytes)
        size_bytes = len(content_bytes)

        # Prepare S3 metadata
        s3_metadata = {
            'checksum-sha256': checksum,
            'size-bytes': str(size_bytes),
            'uploaded-at': datetime.utcnow().isoformat()
        }
        if metadata:
            # S3 metadata values must be strings
            for key, value in metadata.items():
                s3_metadata[f'custom-{key}'] = str(value)

        key = self._get_key(path)

        try:
            async with self._session.create_client('s3', **self._config) as s3_client:
                # Use multipart upload for large files
                if size_bytes > self.MULTIPART_THRESHOLD:
                    await self._multipart_upload(
                        s3_client, key, content_bytes, content_type, s3_metadata
                    )
                else:
                    # Standard single-part upload
                    put_args = {
                        'Bucket': self.bucket,
                        'Key': key,
                        'Body': content_bytes,
                        'Metadata': s3_metadata
                    }
                    if content_type:
                        put_args['ContentType'] = content_type

                    await s3_client.put_object(**put_args)

        except AsyncClientError as e:
            raise RuntimeError(f"Failed to upload to S3: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to upload to S3: {e}")

        return ArtifactRef(
            path=path,
            checksum=checksum,
            size_bytes=size_bytes,
            content_type=content_type,
            metadata=metadata
        )

    async def _multipart_upload(
        self,
        s3_client,
        key: str,
        content_bytes: bytes,
        content_type: Optional[str],
        metadata: Dict[str, str]
    ) -> None:
        """Perform multipart upload for large files."""
        # Initiate multipart upload
        create_args = {
            'Bucket': self.bucket,
            'Key': key,
            'Metadata': metadata
        }
        if content_type:
            create_args['ContentType'] = content_type

        response = await s3_client.create_multipart_upload(**create_args)
        upload_id = response['UploadId']

        try:
            # Upload parts
            parts = []
            part_number = 1
            offset = 0

            while offset < len(content_bytes):
                # Calculate chunk size for this part
                chunk_size = min(self.MULTIPART_CHUNK_SIZE, len(content_bytes) - offset)
                chunk = content_bytes[offset:offset + chunk_size]

                # Upload this part
                part_response = await s3_client.upload_part(
                    Bucket=self.bucket,
                    Key=key,
                    PartNumber=part_number,
                    UploadId=upload_id,
                    Body=chunk
                )

                parts.append({
                    'ETag': part_response['ETag'],
                    'PartNumber': part_number
                })

                offset += chunk_size
                part_number += 1

            # Complete multipart upload
            await s3_client.complete_multipart_upload(
                Bucket=self.bucket,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={'Parts': parts}
            )

        except Exception as e:
            # Abort multipart upload on failure
            try:
                await s3_client.abort_multipart_upload(
                    Bucket=self.bucket,
                    Key=key,
                    UploadId=upload_id
                )
            except Exception:
                pass  # Best effort cleanup
            raise e

    async def get_url(
        self,
        path: str,
        signed: bool = True,
        ttl_seconds: int = 3600
    ) -> str:
        """Get S3 URL (signed or public)."""
        key = self._get_key(path)

        if signed:
            try:
                async with self._session.create_client('s3', **self._config) as s3_client:
                    return s3_client.generate_presigned_url(
                        'get_object',
                        Params={'Bucket': self.bucket, 'Key': key},
                        ExpiresIn=ttl_seconds
                    )
            except Exception as e:
                raise RuntimeError(f"Failed to generate signed URL: {e}")
        else:
            # Return public URL (bucket must allow public access)
            endpoint_url = self._config.get('endpoint_url')
            if endpoint_url:
                # Custom endpoint
                return f"{endpoint_url.rstrip('/')}/{self.bucket}/{key}"
            else:
                # AWS S3
                return f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{key}"

    async def delete(self, path: str) -> bool:
        """Delete artifact from S3."""
        key = self._get_key(path)

        try:
            async with self._session.create_client('s3', **self._config) as s3_client:
                # Check if object exists first since delete_object is idempotent
                try:
                    await s3_client.head_object(Bucket=self.bucket, Key=key)
                except AsyncClientError as e:
                    if e.response['Error']['Code'] in ['404', 'NoSuchKey']:
                        return False  # Object doesn't exist
                    # Re-raise head_object errors (access denied, etc.)
                    raise RuntimeError(f"Failed to check S3 object before deletion: {e}")

                # Object exists, proceed with deletion
                await s3_client.delete_object(Bucket=self.bucket, Key=key)
                return True
        except AsyncClientError as e:
            raise RuntimeError(f"Failed to delete from S3: {e}")
        except Exception as e:
            # Re-raise other exceptions (network errors, access denials, etc.)
            raise RuntimeError(f"Failed to delete from S3: {e}")

    async def exists(self, path: str) -> bool:
        """Check if artifact exists in S3."""
        key = self._get_key(path)

        try:
            async with self._session.create_client('s3', **self._config) as s3_client:
                await s3_client.head_object(Bucket=self.bucket, Key=key)
                return True
        except AsyncClientError as e:
            if e.response['Error']['Code'] in ['404', 'NoSuchKey']:
                return False
            # Re-raise other errors (access denied, network issues, etc.)
            raise RuntimeError(f"Failed to check S3 object existence: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to check S3 object existence: {e}")

    async def get_metadata(self, path: str) -> Optional[Dict[str, Any]]:
        """Get metadata from S3 object."""
        key = self._get_key(path)

        try:
            async with self._session.create_client('s3', **self._config) as s3_client:
                response = await s3_client.head_object(Bucket=self.bucket, Key=key)
                s3_metadata = response.get('Metadata', {})

                # Convert S3 metadata back to regular metadata
                metadata = {}
                for metadata_key, value in s3_metadata.items():
                    if metadata_key.startswith('custom-'):
                        metadata[metadata_key[7:]] = value  # Remove 'custom-' prefix
                    else:
                        metadata[metadata_key] = value

                return metadata
        except AsyncClientError as e:
            if e.response['Error']['Code'] in ['404', 'NoSuchKey']:
                return None
            # Re-raise other errors (access denied, network issues, etc.)
            raise RuntimeError(f"Failed to get S3 metadata: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to get S3 metadata: {e}")


def create_artifact_store(
    backend: str = "local",
    **kwargs
) -> ArtifactStore:
    """Factory function to create artifact store backends.

    Args:
        backend: Storage backend type ("local" or "s3")
        **kwargs: Backend-specific configuration

    Returns:
        Configured ArtifactStore instance
    """
    if backend.lower() == "local":
        return LocalArtifactStore(**kwargs)
    elif backend.lower() == "s3":
        return S3ArtifactStore(**kwargs)
    else:
        raise ValueError(f"Unsupported storage backend: {backend}")


# Default configuration from environment variables
def create_default_artifact_store() -> ArtifactStore:
    """Create artifact store from environment configuration."""
    backend = os.getenv("ARTIFACT_STORAGE_BACKEND", "local").lower()

    if backend == "local":
        base_path = os.getenv("ARTIFACT_STORAGE_PATH", "./artifacts")
        return LocalArtifactStore(base_path=base_path)

    elif backend == "s3":
        return S3ArtifactStore(
            bucket=os.getenv("ARTIFACT_S3_BUCKET", "tag-sentinel-artifacts"),
            prefix=os.getenv("ARTIFACT_S3_PREFIX", ""),
            region=os.getenv("ARTIFACT_S3_REGION", "us-east-1"),
            endpoint_url=os.getenv("ARTIFACT_S3_ENDPOINT_URL"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )

    else:
        raise ValueError(f"Unsupported storage backend: {backend}")