"""OpenTelemetry tracing utilities for the kb-rag pipeline.

Provides a tracer provider initializer, a helper to read the current trace id,
and a decorator / context manager for creating spans.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Iterator, TypeVar

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

F = TypeVar("F", bound=Callable[..., Any])

_INITIALIZED: bool = False


def init_tracer(service_name: str = "kb-rag-api") -> TracerProvider:
    """Initialize and register a global :class:`TracerProvider`.

    Args:
        service_name: Service name reported on every span.

    Returns:
        The configured :class:`TracerProvider`.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return trace.get_tracer_provider()  # type: ignore[return-value]

    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name})
    )
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _INITIALIZED = True
    return provider


def get_current_trace_id() -> str | None:
    """Return the current span's trace id as a zero-padded hex string.

    Returns:
        The 32-char hex trace id, or ``None`` when no valid span is active.
    """
    span = trace.get_current_span()
    context = span.get_span_context()
    if context is not None and context.is_valid:
        return format(context.trace_id, "032x")
    return None


@contextmanager
def start_span(name: str) -> Iterator[Any]:
    """Context manager that starts a child span.

    Args:
        name: Span name.

    Yields:
        The active span.
    """
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span(name) as span:
        yield span


def trace_span(name: str) -> Callable[[F], F]:
    """Decorator wrapping a callable in a span named ``name``.

    Args:
        name: Span name.

    Returns:
        A decorator.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(name):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
