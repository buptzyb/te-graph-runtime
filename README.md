# te-graph-runtime

TE-compatible CUDA graph callable runtime with optional TransformerEngine
integration and a PyTorch-only runtime path.

```python
from te_graph_runtime import make_graphed_callables
```

The public entrypoint intentionally follows TransformerEngine `v2.16`'s
`make_graphed_callables` API, with the CUDA graph parameter-gradient lifetime
fix from NVIDIA/TransformerEngine PR #2937 and `capture_time_hooks` from PR
#2831 carried on top. The implementation does not delegate to
`transformer_engine.pytorch.graph.make_graphed_callables` or to
`torch.cuda.make_graphed_callables`; it captures and replays graphs directly with
PyTorch CUDA graph and autograd primitives.

## Install

Editable checkout:

```bash
pip install -e .
```

With optional TransformerEngine support:

```bash
pip install -e '.[te]'
```

Importing `te_graph_runtime` is lazy: it does not import `torch` or
`transformer_engine` until `make_graphed_callables` is called.

## Runtime modes

- **No TransformerEngine installed:** generic PyTorch `nn.Module` graphing works,
  including `sample_kwargs`, `_order`, `_num_layers_per_chunk`, warmup hooks,
  `capture_time_hooks`, graph reset, stream/event replay kwargs, structural
  `backward_dw` support, and PR #2937's `_clone_param_grads_on_return` lifetime
  policy. FP8/TE-specific
  options fail fast with an actionable error.
- **TransformerEngine installed:** TE internals are used for FP8/FP4 recipes,
  amax reduction, TE module state save/restore, TE RNG tracker state, and TE
  module `backward_dw` handling. Tests compare behavior with TE's upstream
  implementation.

## Examples

- `examples/custom_framework_minimal.py` - wrap a custom framework block and use
  keyword tensor inputs.
- `examples/custom_framework_pipeline.py` - integrate an interleaved pipeline
  schedule with `_order` and optional static-buffer reuse.
- `examples/optional_te_fp8.py` - use one code path that enables FP8 only when
  TransformerEngine and FP8-capable hardware are present.

See `docs/custom_framework_integration.md` for integration patterns, failure
modes, and test commands.

## Test commands

CPU/local static tests:

```bash
PYTHONPATH=src python -m pytest -q
```

GPU tests without requiring TransformerEngine:

```bash
PYTHONPATH=src python -m pytest -q tests/test_cuda_runtime.py
```

GPU tests with TransformerEngine parity checks:

```bash
PYTHONPATH=src python -m pytest -q tests/test_te_parity.py
```

See `UPSTREAM.md` for the pinned upstream source and update workflow.
