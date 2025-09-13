"""Alert dispatching framework for rule evaluation notifications.

This package provides a flexible alert dispatching system that can send
notifications through multiple channels (webhook, email, etc.) when rule
evaluations fail or meet specific trigger conditions.
"""

from .base import (
    # Alert status and severity enums
    AlertStatus,
    AlertSeverity,
    AlertTrigger,
    
    # Core alert classes
    AlertContext,
    AlertPayload,
    AlertTemplate,
    AlertDispatchResult,
    
    # Base dispatcher framework
    BaseAlertDispatcher,
    AlertDispatcherRegistry,
    
    # Registry and decorators
    dispatcher_registry,
    register_dispatcher,
)

# Import concrete dispatcher implementations to register them
from . import webhook
from . import email

__all__ = [
    # Alert status and severity enums
    'AlertStatus',
    'AlertSeverity', 
    'AlertTrigger',
    
    # Core alert classes
    'AlertContext',
    'AlertPayload',
    'AlertTemplate',
    'AlertDispatchResult',
    
    # Base dispatcher framework
    'BaseAlertDispatcher',
    'AlertDispatcherRegistry',
    
    # Registry and decorators
    'dispatcher_registry',
    'register_dispatcher',
]