"""
Logging Middleware

Provides logging capabilities for graph node execution.
"""

import logging
import time
from typing import Callable, Dict, Any
from functools import wraps

from data_agent.middleware.base import NodeMiddleware


logger = logging.getLogger("data_agent.nodes")


class LoggingMiddleware:
    """
    Middleware that logs node entry, exit, and execution time.
    
    Example:
        middleware = LoggingMiddleware(log_level=logging.DEBUG)
        wrapped_node = middleware.wrap(my_node)
    """
    
    def __init__(
        self,
        log_level: int = logging.INFO,
        log_state_keys: bool = True,
        log_timing: bool = True,
    ):
        """
        Initialize logging middleware.
        
        Args:
            log_level: Logging level for messages
            log_state_keys: Whether to log state keys on entry/exit
            log_timing: Whether to log execution time
        """
        self.log_level = log_level
        self.log_state_keys = log_state_keys
        self.log_timing = log_timing
    
    def wrap(self, node_fn: Callable) -> Callable:
        """Wrap a node function with logging."""
        node_name = getattr(node_fn, '__name__', str(node_fn))
        
        @wraps(node_fn)
        async def async_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            # Log entry
            if self.log_state_keys:
                state_keys = list(state.keys()) if state else []
                logger.log(self.log_level, f"[{node_name}] ENTER - state keys: {state_keys}")
            else:
                logger.log(self.log_level, f"[{node_name}] ENTER")
            
            start_time = time.time()
            
            try:
                result = await node_fn(state)
                
                elapsed = time.time() - start_time
                
                # Log exit
                if self.log_state_keys:
                    result_keys = list(result.keys()) if result else []
                    msg = f"[{node_name}] EXIT - output keys: {result_keys}"
                else:
                    msg = f"[{node_name}] EXIT"
                
                if self.log_timing:
                    msg += f" ({elapsed:.3f}s)"
                
                logger.log(self.log_level, msg)
                
                return result
                
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"[{node_name}] ERROR after {elapsed:.3f}s: {str(e)}")
                raise
        
        @wraps(node_fn)
        def sync_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            # Log entry
            if self.log_state_keys:
                state_keys = list(state.keys()) if state else []
                logger.log(self.log_level, f"[{node_name}] ENTER - state keys: {state_keys}")
            else:
                logger.log(self.log_level, f"[{node_name}] ENTER")
            
            start_time = time.time()
            
            try:
                result = node_fn(state)
                
                elapsed = time.time() - start_time
                
                # Log exit
                if self.log_state_keys:
                    result_keys = list(result.keys()) if result else []
                    msg = f"[{node_name}] EXIT - output keys: {result_keys}"
                else:
                    msg = f"[{node_name}] EXIT"
                
                if self.log_timing:
                    msg += f" ({elapsed:.3f}s)"
                
                logger.log(self.log_level, msg)
                
                return result
                
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"[{node_name}] ERROR after {elapsed:.3f}s: {str(e)}")
                raise
        
        # Check if the original function is async
        import asyncio
        if asyncio.iscoroutinefunction(node_fn):
            return async_wrapper
        return sync_wrapper
