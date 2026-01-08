"""
Metrics Middleware

Provides metrics collection for graph node execution.
"""

import time
from typing import Callable, Dict, Any, Optional
from functools import wraps
from dataclasses import dataclass, field
from collections import defaultdict

from data_agent.middleware.base import NodeMiddleware


@dataclass
class NodeMetrics:
    """Metrics for a single node."""
    
    call_count: int = 0
    total_time: float = 0.0
    error_count: int = 0
    last_execution_time: float = 0.0
    
    @property
    def avg_time(self) -> float:
        """Average execution time."""
        if self.call_count == 0:
            return 0.0
        return self.total_time / self.call_count


class MetricsCollector:
    """
    Collects and stores metrics for graph execution.
    
    Example:
        collector = MetricsCollector()
        middleware = MetricsMiddleware(collector)
        
        # After execution
        print(collector.get_summary())
    """
    
    def __init__(self):
        self._metrics: Dict[str, NodeMetrics] = defaultdict(NodeMetrics)
    
    def record_execution(
        self,
        node_name: str,
        execution_time: float,
        error: bool = False,
    ) -> None:
        """Record a node execution."""
        metrics = self._metrics[node_name]
        metrics.call_count += 1
        metrics.total_time += execution_time
        metrics.last_execution_time = execution_time
        if error:
            metrics.error_count += 1
    
    def get_metrics(self, node_name: str) -> NodeMetrics:
        """Get metrics for a specific node."""
        return self._metrics.get(node_name, NodeMetrics())
    
    def get_all_metrics(self) -> Dict[str, NodeMetrics]:
        """Get all collected metrics."""
        return dict(self._metrics)
    
    def get_summary(self) -> str:
        """Get a formatted summary of all metrics."""
        lines = ["Node Metrics Summary:", "=" * 50]
        
        for name, metrics in sorted(self._metrics.items()):
            lines.append(
                f"{name}: calls={metrics.call_count}, "
                f"avg={metrics.avg_time:.3f}s, "
                f"total={metrics.total_time:.3f}s, "
                f"errors={metrics.error_count}"
            )
        
        return "\n".join(lines)
    
    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics.clear()


# Global metrics collector
_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


class MetricsMiddleware:
    """
    Middleware that collects execution metrics for nodes.
    
    Example:
        middleware = MetricsMiddleware()
        wrapped_node = middleware.wrap(my_node)
        
        # After execution
        print(get_metrics_collector().get_summary())
    """
    
    def __init__(self, collector: Optional[MetricsCollector] = None):
        """
        Initialize metrics middleware.
        
        Args:
            collector: Optional metrics collector, uses global if not provided
        """
        self.collector = collector or get_metrics_collector()
    
    def wrap(self, node_fn: Callable) -> Callable:
        """Wrap a node function with metrics collection."""
        node_name = getattr(node_fn, '__name__', str(node_fn))
        
        @wraps(node_fn)
        async def async_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            start_time = time.time()
            error = False
            
            try:
                result = await node_fn(state)
                return result
            except Exception as e:
                error = True
                raise
            finally:
                elapsed = time.time() - start_time
                self.collector.record_execution(node_name, elapsed, error)
        
        @wraps(node_fn)
        def sync_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            start_time = time.time()
            error = False
            
            try:
                result = node_fn(state)
                return result
            except Exception as e:
                error = True
                raise
            finally:
                elapsed = time.time() - start_time
                self.collector.record_execution(node_name, elapsed, error)
        
        # Check if the original function is async
        import asyncio
        if asyncio.iscoroutinefunction(node_fn):
            return async_wrapper
        return sync_wrapper
