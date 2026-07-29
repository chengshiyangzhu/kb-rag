"""Structured logging configuration built on structlog.

Produces JSON-formatted log events and bridges structlog with the standard
``logging`` module so third-party libraries are captured. A ``trace_id`` can be
bound to the current context so that every log line within a request span is
correlatable.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

import structlog

# ContextVar holding the active trace_id for the current async/task context.
_trace_id_var: ContextVar[str | None] = ContextVar("kb_rag_trace_id", default=None)

# Guard against repeated configuration (e.g. on re-import in tests).
_CONFIGURED: bool = False


def configure_logging(level: str = "INFO") -> None:
    """Initialize structlog with JSON rendering and bridge to standard logging.

    Args:
        level: Log level name (e.g. ``"INFO"``, ``"DEBUG"``).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_level = getattr(logging, level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _inject_trace_id,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Bridge: route stdlib logging through structlog so library logs are JSON too.
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[logging.StreamHandler()],
    )

    _CONFIGURED = True


def _inject_trace_id(
    _logger: Any, _method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Processor that injects the current ``trace_id`` into the event dict."""
    trace_id = _trace_id_var.get()
    if trace_id:
        event_dict["trace_id"] = trace_id
    return event_dict


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger.

    Args:
        name: Logical logger name (e.g. module ``__name__``).

    Returns:
        A bound structlog logger. Configures logging on first call.
    """
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name)


@contextmanager
def bind_trace_id(trace_id: str) -> Iterator[str]:
    """Bind a ``trace_id`` to the current context for the duration of the block.

    Args:
        trace_id: Identifier of the active trace/span.

    Yields:
        The provided ``trace_id``.

    Example:
        >>> with bind_trace_id("abc123"):
        ...     get_logger().info("handling request")
    """
    token = _trace_id_var.set(trace_id)
    try:
        yield trace_id
    finally:
        _trace_id_var.reset(token)
