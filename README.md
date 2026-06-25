# te-graph-runtime

TE-compatible CUDA graph callable runtime with optional TransformerEngine
integration and a PyTorch-only fallback path.

The public entrypoint intentionally matches TransformerEngine's
`make_graphed_callables` API:

```python
from te_graph_runtime import make_graphed_callables
```

## Current state

This repository is bootstrapped from TransformerEngine `v2.16`.

- If `transformer_engine` is installed, calls delegate to
  `transformer_engine.pytorch.graph.make_graphed_callables`.
- If `transformer_engine` is not installed, the basic PyTorch
  `torch.cuda.make_graphed_callables` argument subset is supported.
- TE-only options such as FP8 recipes, amax reduction groups, and quantized
  parameter caching require TransformerEngine.
- Extended torch-only behavior such as `sample_kwargs`, `_order`, graph buffer
  reuse, and warmup hooks is planned but not split out yet.

See `UPSTREAM.md` for the pinned upstream source and update workflow.

