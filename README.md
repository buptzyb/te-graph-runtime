# te-graph-runtime

TE-compatible CUDA graph callable runtime with optional TransformerEngine
integration and a PyTorch-only runtime path.

The public entrypoint intentionally matches TransformerEngine's
`make_graphed_callables` API:

```python
from te_graph_runtime import make_graphed_callables
```

## Current state

This repository is bootstrapped from TransformerEngine `v2.16`. The runtime does
not delegate to `transformer_engine.pytorch.graph.make_graphed_callables` or to
`torch.cuda.make_graphed_callables`; it implements capture/replay directly with
PyTorch CUDA graph and autograd primitives.

- Package import is lazy and does not import `torch` or `transformer_engine`.
- If `transformer_engine` internals are available at call time, TE-specific FP8,
  amax, recipe, and module-state behavior is enabled through those internals.
- If `transformer_engine` is not available, generic graphing features remain
  available, while FP8/TE-specific options fail fast with an actionable error.
- The source-compatible API is pinned to the TE `v2.16` signature.

See `UPSTREAM.md` for the pinned upstream source and update workflow.
