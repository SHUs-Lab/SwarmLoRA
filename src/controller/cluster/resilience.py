"""Resilience utilities: retry, circuit breaker, error classification."""
import asyncio
import time
import random
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Callable, TypeVar, Awaitable
import aiohttp

logger = logging.getLogger(__name__)
T = TypeVar('T')

# Error Classification

class ErrorCategory(Enum):
    TRANSIENT = "transient"   # Retry: timeouts, 503, connection refused
    PERMANENT = "permanent"   # Don't retry: 4xx, invalid request
    UNKNOWN = "unknown"       # Retry with caution


def classify_http_status(status: int) -> ErrorCategory:
    """Classify HTTP status code into error category."""
    if status in (408, 429, 500, 502, 503, 504):
        return ErrorCategory.TRANSIENT
    if 400 <= status < 500:
        return ErrorCategory.PERMANENT
    return ErrorCategory.UNKNOWN


def classify_error(exc: Exception) -> ErrorCategory:
    """Classify an exception into error category."""
    if isinstance(exc, asyncio.TimeoutError):
        return ErrorCategory.TRANSIENT
    if isinstance(exc, (aiohttp.ClientConnectorError, aiohttp.ServerDisconnectedError)):
        return ErrorCategory.TRANSIENT
    if isinstance(exc, ConnectionRefusedError):
        return ErrorCategory.TRANSIENT
    if isinstance(exc, OSError) and exc.errno in (111, 113):  # Connection refused, No route
        return ErrorCategory.TRANSIENT
    if hasattr(exc, 'status'):
        return classify_http_status(exc.status)
    return ErrorCategory.UNKNOWN


# Retry with Exponential Backoff

@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: float = 0.1  # 10% jitter


class RetryExecutor:
    """Executes async functions with retry and exponential backoff."""

    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        delay = min(
            self.config.base_delay * (self.config.exponential_base ** attempt),
            self.config.max_delay
        )
        jitter = delay * self.config.jitter * random.random()
        return delay + jitter

    async def execute(
        self,
        func: Callable[[], Awaitable[T]],
        classifier: Callable[[Exception], ErrorCategory] = classify_error,
    ) -> T:
        """Execute func with retry."""
        last_error = None
        for attempt in range(self.config.max_attempts):
            try:
                return await func()
            except Exception as e:
                if classifier(e) == ErrorCategory.PERMANENT:
                    raise
                last_error = e
                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    logger.warning(f"Retry {attempt+1}/{self.config.max_attempts} in {delay:.2f}s: {e}")
                    await asyncio.sleep(delay)
        raise last_error


# Circuit Breaker

class CircuitState(Enum):
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Failing - reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 50   # Failures to open (relaxed to avoid cascade)
    success_threshold: int = 2    # Successes to close from half-open
    open_duration: float = 10.0   # Seconds before half-open
    half_open_max: int = 3        # Max requests in half-open


class CircuitBreaker:
    """Circuit breaker for fault isolation."""

    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_count = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    async def acquire(self) -> bool:
        """Try to acquire permission to make a request."""
        async with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if enough time has passed to try half-open
                if time.time() - self._last_failure_time >= self.config.open_duration:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_count = 0
                    self._success_count = 0
                    logger.info(f"Circuit '{self.name}' -> half-open")
                else:
                    return False

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_count >= self.config.half_open_max:
                    return False
                self._half_open_count += 1

            return True

    async def record_success(self):
        """Record a successful request."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info(f"Circuit '{self.name}' -> CLOSED (recovered)")
            else:
                # Reset failure count on success in closed state
                self._failure_count = 0

    async def record_failure(self):
        """Record a failed request."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open reopens the circuit
                self._state = CircuitState.OPEN
                logger.warning(f"Circuit '{self.name}' -> OPEN (half-open failed)")
            elif self._failure_count >= self.config.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(f"Circuit '{self.name}' -> OPEN (threshold reached: {self._failure_count})")

    def get_status(self) -> dict:
        """Get circuit breaker status for monitoring."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_time": self._last_failure_time,
        }
