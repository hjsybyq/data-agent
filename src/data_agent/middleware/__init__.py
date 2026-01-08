"""
Middleware Module

Provides extensible middleware for wrapping graph nodes.
"""

from data_agent.middleware.base import (
    NodeMiddleware,
    apply_middleware,
    middleware_chain,
)
from data_agent.middleware.logging import LoggingMiddleware
from data_agent.middleware.metrics import (
    MetricsMiddleware,
    MetricsCollector,
    NodeMetrics,
    get_metrics_collector,
)

__all__ = [
    # Base
    "NodeMiddleware",
    "apply_middleware",
    "middleware_chain",
    # Logging
    "LoggingMiddleware",
    # Metrics
    "MetricsMiddleware",
    "MetricsCollector",
    "NodeMetrics",
    "get_metrics_collector",
]
