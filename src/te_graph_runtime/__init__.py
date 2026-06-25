"""TE-compatible CUDA graph callable runtime."""

from .graph import (
    UPSTREAM_TE_COMMIT,
    UPSTREAM_TE_GRAPH_PATH,
    UPSTREAM_TE_VERSION,
    make_graphed_callables,
)

__all__ = [
    "UPSTREAM_TE_COMMIT",
    "UPSTREAM_TE_GRAPH_PATH",
    "UPSTREAM_TE_VERSION",
    "make_graphed_callables",
]

