"""
Middleware Base Module

Defines the protocol for node middleware and utility functions
for wrapping graph nodes with middleware chains.
"""

from typing import Callable, Dict, Any, Sequence, Protocol, runtime_checkable
from functools import wraps


@runtime_checkable
class NodeMiddleware(Protocol):
    """
    Protocol for node middleware.
    
    Middleware wraps node functions to add cross-cutting concerns
    like logging, metrics, caching, etc.
    """
    
    def wrap(self, node_fn: Callable) -> Callable:
        """
        Wrap a node function with middleware logic.
        
        Args:
            node_fn: The original node function
            
        Returns:
            Wrapped function with middleware applied
        """
        ...


def apply_middleware(
    node_fn: Callable,
    middleware: Sequence[NodeMiddleware],
) -> Callable:
    """
    Apply a chain of middleware to a node function.
    
    Middleware is applied in order, with the first middleware
    being the outermost wrapper.
    
    Args:
        node_fn: The original node function
        middleware: Sequence of middleware to apply
        
    Returns:
        Wrapped function with all middleware applied
        
    Example:
        wrapped = apply_middleware(
            my_node,
            [LoggingMiddleware(), MetricsMiddleware()]
        )
    """
    wrapped = node_fn
    # Apply in reverse order so first middleware is outermost
    for mw in reversed(middleware):
        wrapped = mw.wrap(wrapped)
    return wrapped


def middleware_chain(*middleware: NodeMiddleware) -> Callable:
    """
    Create a decorator that applies middleware to a node function.
    
    Example:
        @middleware_chain(LoggingMiddleware(), MetricsMiddleware())
        async def my_node(state):
            ...
    """
    def decorator(node_fn: Callable) -> Callable:
        return apply_middleware(node_fn, middleware)
    return decorator
