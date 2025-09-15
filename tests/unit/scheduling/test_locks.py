"""Unit tests for distributed concurrency control system."""

import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from app.scheduling.locks import (
    ConcurrencyManager,
    LockInfo,
    LockError,
    LockTimeoutError,
    LockBackendError,
    InMemoryLockBackend,
    create_concurrency_manager
)


class TestLockInfo:
    """Test LockInfo functionality."""

    def test_lock_info_creation(self):
        """Test LockInfo creation and basic properties."""
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=3600)

        lock_info = LockInfo(
            key="test-key",
            owner_id="owner-123",
            acquired_at=now,
            expires_at=expires_at,
            metadata={"site_id": "test", "environment": "prod"}
        )

        assert lock_info.key == "test-key"
        assert lock_info.owner_id == "owner-123"
        assert lock_info.acquired_at == now
        assert lock_info.expires_at == expires_at
        assert lock_info.metadata["site_id"] == "test"

    def test_lock_expiration_check(self):
        """Test lock expiration detection."""
        now = datetime.now(timezone.utc)

        # Expired lock
        expired_lock = LockInfo(
            key="expired",
            owner_id="owner-1",
            acquired_at=now - timedelta(hours=2),
            expires_at=now - timedelta(hours=1),
            metadata={}
        )
        assert expired_lock.is_expired is True

        # Active lock
        active_lock = LockInfo(
            key="active",
            owner_id="owner-2",
            acquired_at=now - timedelta(minutes=30),
            expires_at=now + timedelta(minutes=30),
            metadata={}
        )
        assert active_lock.is_expired is False

    def test_time_remaining(self):
        """Test time remaining calculation."""
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=1800)  # 30 minutes

        lock_info = LockInfo(
            key="test",
            owner_id="owner",
            acquired_at=now,
            expires_at=expires_at,
            metadata={}
        )

        remaining = lock_info.time_remaining
        assert remaining.total_seconds() > 1700  # Should be close to 30 minutes
        assert remaining.total_seconds() < 1900

    def test_lock_serialization(self):
        """Test LockInfo serialization and deserialization."""
        now = datetime.now(timezone.utc)
        original = LockInfo(
            key="serialize-test",
            owner_id="owner-456",
            acquired_at=now,
            expires_at=now + timedelta(seconds=3600),
            metadata={"test": "value"}
        )

        # Serialize
        data = original.to_dict()
        assert isinstance(data, dict)
        assert data["key"] == "serialize-test"
        assert data["owner_id"] == "owner-456"

        # Deserialize
        restored = LockInfo.from_dict(data)
        assert restored.key == original.key
        assert restored.owner_id == original.owner_id
        assert restored.acquired_at == original.acquired_at
        assert restored.expires_at == original.expires_at
        assert restored.metadata == original.metadata


class TestInMemoryLockBackend:
    """Test InMemoryLockBackend functionality."""

    @pytest.fixture
    def backend(self):
        """Create an in-memory lock backend for testing."""
        return InMemoryLockBackend()

    @pytest.mark.asyncio
    async def test_acquire_and_release_lock(self, backend):
        """Test basic lock acquisition and release."""
        key = "test-lock"
        owner_id = str(uuid4())

        # Acquire lock
        acquired = await backend.acquire_lock(key, owner_id, timeout_seconds=300, metadata={})
        assert acquired is True

        # Get lock info
        lock_info = await backend.get_lock_info(key)
        assert lock_info is not None
        assert lock_info.key == key
        assert lock_info.owner_id == owner_id
        assert not lock_info.is_expired

        # Release lock
        released = await backend.release_lock(key, owner_id)
        assert released is True

        # Try to release again (should fail)
        released_again = await backend.release_lock(key, owner_id)
        assert released_again is False

    @pytest.mark.asyncio
    async def test_lock_conflict(self, backend):
        """Test that two owners cannot acquire the same lock."""
        key = "conflict-test"
        owner1 = str(uuid4())
        owner2 = str(uuid4())

        # First owner acquires lock
        acquired1 = await backend.acquire_lock(key, owner1, timeout_seconds=300, metadata={})
        assert acquired1 is True

        # Second owner attempts to acquire same lock
        acquired2 = await backend.acquire_lock(key, owner2, timeout_seconds=300, metadata={})
        assert acquired2 is False  # Should fail

        # First owner releases lock
        released = await backend.release_lock(key, owner1)
        assert released is True

        # Now second owner can acquire
        acquired3 = await backend.acquire_lock(key, owner2, timeout_seconds=300, metadata={})
        assert acquired3 is True

        # Verify ownership
        lock_info = await backend.get_lock_info(key)
        assert lock_info is not None
        assert lock_info.owner_id == owner2

    @pytest.mark.asyncio
    async def test_lock_extension(self, backend):
        """Test extending lock expiration."""
        key = "extend-test"
        owner_id = str(uuid4())

        # Acquire lock with short timeout
        acquired = await backend.acquire_lock(key, owner_id, timeout_seconds=60, metadata={})
        assert acquired is True

        # Get initial lock info
        lock_info = await backend.get_lock_info(key)
        assert lock_info is not None
        original_expires = lock_info.expires_at

        # Extend lock
        extended = await backend.extend_lock(key, owner_id, additional_seconds=300)
        assert extended is True

        # Check that expiration was extended
        updated_info = await backend.get_lock_info(key)
        assert updated_info is not None
        assert updated_info.expires_at > original_expires

    @pytest.mark.asyncio
    async def test_get_lock_info(self, backend):
        """Test retrieving lock information."""
        key = "info-test"
        owner_id = str(uuid4())

        # No lock initially
        info = await backend.get_lock_info(key)
        assert info is None

        # Acquire lock
        acquired = await backend.acquire_lock(key, owner_id, timeout_seconds=300, metadata={})
        assert acquired is True

        # Get lock info
        retrieved_info = await backend.get_lock_info(key)
        assert retrieved_info is not None
        assert retrieved_info.key == key
        assert retrieved_info.owner_id == owner_id

    @pytest.mark.asyncio
    async def test_cleanup_expired_locks(self, backend):
        """Test cleanup of expired locks."""
        key = "cleanup-test"
        owner_id = str(uuid4())

        # Acquire lock with very short timeout
        acquired = await backend.acquire_lock(key, owner_id, timeout_seconds=1, metadata={})
        assert acquired is True

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Cleanup expired locks
        cleaned_count = await backend.cleanup_expired_locks()
        assert cleaned_count == 1

        # Lock should be gone
        info = await backend.get_lock_info(key)
        assert info is None

    @pytest.mark.asyncio
    async def test_list_locks(self, backend):
        """Test listing active locks."""
        # Initially empty
        locks = await backend.list_locks()
        assert len(locks) == 0

        # Add some locks
        acquired1 = await backend.acquire_lock("lock1", str(uuid4()), timeout_seconds=300, metadata={})
        acquired2 = await backend.acquire_lock("lock2", str(uuid4()), timeout_seconds=300, metadata={})
        assert acquired1 is True
        assert acquired2 is True

        # List all locks
        locks = await backend.list_locks()
        assert len(locks) == 2

        lock_keys = {lock.key for lock in locks}
        assert "lock1" in lock_keys
        assert "lock2" in lock_keys

    @pytest.mark.asyncio
    async def test_health_check(self, backend):
        """Test health check functionality."""
        health = await backend.health_check()
        assert isinstance(health, dict)
        assert "backend_type" in health
        assert health["backend_type"] == "InMemoryLockBackend"


class TestConcurrencyManager:
    """Test ConcurrencyManager functionality."""

    @pytest.fixture
    async def manager(self):
        """Create a ConcurrencyManager for testing."""
        backend = InMemoryLockBackend()
        manager = ConcurrencyManager(backend, default_timeout_seconds=300, metadata={})
        await manager.start()
        yield manager
        await manager.stop()

    @pytest.mark.asyncio
    async def test_acquire_site_lock(self, manager):
        """Test acquiring and releasing site-environment locks."""
        site_id = "test-site"
        environment = "production"
        owner_id = str(uuid4())

        # Acquire lock
        async with manager.acquire_lock(site_id, environment, owner_id) as lock_info:
            assert lock_info is not None
            assert lock_info.key == f"audit-lock:test-site:production"
            assert lock_info.owner_id == owner_id

            # Check lock status
            is_locked = await manager.is_locked(site_id, environment)
            assert is_locked is True

        # Lock should be released after context
        is_locked = await manager.is_locked(site_id, environment)
        assert is_locked is False

    @pytest.mark.asyncio
    async def test_concurrent_lock_conflict(self, manager):
        """Test that concurrent lock acquisition fails appropriately."""
        site_id = "conflict-site"
        environment = "staging"
        owner1 = str(uuid4())
        owner2 = str(uuid4())

        # First manager acquires lock
        async with manager.acquire_lock(site_id, environment, owner1):
            # Second manager should timeout trying to acquire same lock
            with pytest.raises(LockTimeoutError):
                async with manager.acquire_lock(
                    site_id,
                    environment,
                    owner2,
                    timeout_seconds=1  # Short timeout for test
                ):
                    pass  # Should never reach here

    @pytest.mark.asyncio
    async def test_lock_status_information(self, manager):
        """Test retrieving lock status information."""
        site_id = "status-test"
        environment = "development"
        owner_id = str(uuid4())

        # No lock initially
        status = await manager.get_lock_status(site_id, environment)
        assert status is None

        # Acquire lock and check status
        async with manager.acquire_lock(site_id, environment, owner_id):
            status = await manager.get_lock_status(site_id, environment)
            assert status is not None
            assert status.owner_id == owner_id
            assert not status.is_expired

    @pytest.mark.asyncio
    async def test_list_active_locks(self, manager):
        """Test listing all active locks."""
        # Initially no locks
        locks = await manager.list_active_locks()
        assert len(locks) == 0

        # Acquire multiple locks
        owner1 = str(uuid4())
        owner2 = str(uuid4())

        async with manager.acquire_lock("site1", "prod", owner1):
            async with manager.acquire_lock("site2", "staging", owner2):
                # Should have two active locks
                locks = await manager.list_active_locks()
                assert len(locks) == 2

                # Check lock details
                lock_keys = {lock.key for lock in locks}
                assert "audit-lock:site1:prod" in lock_keys
                assert "audit-lock:site2:staging" in lock_keys

    @pytest.mark.asyncio
    async def test_force_release_lock(self, manager):
        """Test force releasing locks (admin operation)."""
        site_id = "force-release"
        environment = "prod"
        owner_id = str(uuid4())

        # Acquire lock
        async with manager.acquire_lock(site_id, environment, owner_id):
            # Verify lock is active
            assert await manager.is_locked(site_id, environment) is True

            # Force release from outside
            released = await manager.force_release_lock(site_id, environment)
            assert released is True

            # Lock should be gone
            assert await manager.is_locked(site_id, environment) is False

    @pytest.mark.asyncio
    async def test_health_check(self, manager):
        """Test manager health check."""
        health = await manager.health_check()
        assert isinstance(health, dict)
        assert "backend_health" in health
        assert "default_timeout_seconds" in health
        assert health["default_timeout_seconds"] == 300

    @pytest.mark.asyncio
    async def test_context_manager_error_handling(self, manager):
        """Test that locks are released even if errors occur."""
        site_id = "error-test"
        environment = "test"
        owner_id = str(uuid4())

        # Acquire lock and raise exception
        with pytest.raises(ValueError):
            async with manager.acquire_lock(site_id, environment, owner_id):
                # Verify lock is active during execution
                assert await manager.is_locked(site_id, environment) is True
                raise ValueError("Test error")

        # Lock should be released despite the error
        assert await manager.is_locked(site_id, environment) is False


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.mark.asyncio
    async def test_create_concurrency_manager_inmemory(self):
        """Test creating in-memory concurrency manager."""
        manager = await create_concurrency_manager(
            backend_type="memory",
            default_timeout_seconds=1800
        )

        assert isinstance(manager, ConcurrencyManager)
        assert manager.default_timeout_seconds == 1800

        # Test basic functionality
        await manager.start()
        try:
            owner_id = str(uuid4())
            async with manager.acquire_lock("test", "env", owner_id) as lock:
                assert lock is not None
        finally:
            await manager.stop()

    @pytest.mark.asyncio
    async def test_create_concurrency_manager_redis_unavailable(self):
        """Test creating Redis manager when Redis is unavailable."""
        # This should fall back to memory backend
        manager = await create_concurrency_manager(
            backend_type="redis",
            redis_url="redis://nonexistent:6379"
        )

        # Should fall back to in-memory backend
        assert isinstance(manager, ConcurrencyManager)

        await manager.start()
        try:
            # Test that it works
            owner_id = str(uuid4())
            async with manager.acquire_lock("fallback", "test", owner_id) as lock:
                assert lock is not None
        finally:
            await manager.stop()


class TestErrorConditions:
    """Test error conditions and edge cases."""

    @pytest.mark.asyncio
    async def test_double_release(self):
        """Test releasing a lock twice."""
        backend = InMemoryLockBackend()
        key = "double-release"
        owner_id = str(uuid4())

        # Acquire and release
        acquired = await backend.acquire_lock(key, owner_id, timeout_seconds=300, metadata={})
        assert acquired is True

        first_release = await backend.release_lock(key, owner_id)
        assert first_release is True

        # Second release should return False (lock doesn't exist)
        second_release = await backend.release_lock(key, owner_id)
        assert second_release is False

    @pytest.mark.asyncio
    async def test_wrong_owner_release(self):
        """Test releasing lock with wrong owner ID."""
        backend = InMemoryLockBackend()
        key = "wrong-owner"
        owner1 = str(uuid4())
        owner2 = str(uuid4())

        # Owner 1 acquires lock
        acquired = await backend.acquire_lock(key, owner1, timeout_seconds=300, metadata={})
        assert acquired is True

        # Owner 2 tries to release (should fail)
        released = await backend.release_lock(key, owner2)
        assert released is False

        # Lock should still exist
        info = await backend.get_lock_info(key)
        assert info is not None
        assert info.owner_id == owner1

    @pytest.mark.asyncio
    async def test_extend_nonexistent_lock(self):
        """Test extending a lock that doesn't exist."""
        backend = InMemoryLockBackend()

        extended = await backend.extend_lock("nonexistent", "owner", 300)
        assert extended is False

    @pytest.mark.asyncio
    async def test_manager_operations_before_start(self):
        """Test that manager operations work correctly before start() is called."""
        backend = InMemoryLockBackend()
        manager = ConcurrencyManager(backend, default_timeout_seconds=300, metadata={})

        # Should still work without calling start()
        owner_id = str(uuid4())
        async with manager.acquire_lock("test", "env", owner_id) as lock:
            assert lock is not None