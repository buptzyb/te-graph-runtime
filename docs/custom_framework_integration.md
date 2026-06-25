# Custom Framework Integration Guide

`te-graph-runtime` is meant to be imported where a framework would otherwise use
`transformer_engine.pytorch.make_graphed_callables`.

```python
from te_graph_runtime import make_graphed_callables
```

The callable signature is compatible with TransformerEngine `v2.16`. The package
import is lazy, so framework plugin registration can import it before CUDA,
PyTorch, or TransformerEngine have initialized.

## Case 1: Plain module capture

Use this when a framework has a regular `torch.nn.Module` block and fixed input
shapes during replay.

```python
graphed_block = make_graphed_callables(
    block,
    (sample_hidden_states,),
    allow_unused_input=True,
)
output = graphed_block(hidden_states)
```

Constraints are the same CUDA graph constraints users expect: stable shapes,
stable tensor structure, no module hooks at capture time, and no trainable
buffers.

## Case 2: Framework kwargs

Framework blocks often receive tensor metadata such as masks, positions, routing
maps, or packed-sequence offsets through kwargs. Provide `sample_kwargs` during
capture and pass the same kwarg names during replay.

```python
graphed_attention = make_graphed_callables(
    attention_block,
    (sample_hidden_states,),
    sample_kwargs={"attention_mask": sample_mask, "position_ids": sample_pos},
)
output = graphed_attention(hidden_states, attention_mask=mask, position_ids=pos)
```

All sampled kwargs must be tensor leaves. The replay call must provide every
kwarg key captured in `sample_kwargs`.

## Case 3: Interleaved pipeline schedules

When the framework captures multiple layers under a pipeline schedule, pass the
layers as a tuple and provide `_order`. Positive values are forward chunks;
negative values are backward chunks. This mirrors the TE/Megatron convention.

```python
layer_forwards = make_graphed_callables(
    tuple(layers),
    sample_args,
    _order=[1, 2, 1, 2, -2, -1, -2, -1],
    allow_unused_input=True,
)
```

If non-overlapping microbatches use the same tensor signatures, enable static
buffer reuse:

```python
layer_forwards = make_graphed_callables(
    tuple(layers),
    sample_args,
    _order=order,
    _reuse_graph_input_output_buffers=True,
)
```

Use buffer reuse only when the framework's communication schedule guarantees the
old static outputs and grad inputs are no longer needed. Returned parameter grads
are cloned by default to avoid exposing CUDA graph static buffers. Advanced
frameworks that consume parameter grads before any later replay can pass
`_clone_param_grads_on_return=False` to skip the extra clone, but retained grad
hooks and `.grad` tensors then have non-standard lifetime semantics.

## Case 4: Capture-time framework hooks

Framework hooks registered directly on `torch.nn.Module` are rejected during
graph capture because they may be recorded into the CUDA graph. If a framework
must run non-capturable bookkeeping during warmup and capture, pass the hook
functions explicitly through `capture_time_hooks`. They run outside CUDA graph
capture and are not replayed.

```python
capture_hooks = [{
    "forward_pre_hooks": {0: lambda module, args: prepare(module)},
    "backward_hooks": {1: lambda module, grad_in, grad_out: finalize(module)},
}]
graphed_block = make_graphed_callables(
    block,
    (sample_hidden_states,),
    capture_time_hooks=capture_hooks,
)
```

Hook list entries correspond to callables, not microbatches. Hooks must return
`None`; modifying args, kwargs, outputs, or gradients by returning replacements
is rejected. For hook signatures with kwargs, include the hook ID in
`forward_pre_hooks_with_kwargs` or `forward_hooks_with_kwargs`.

## Case 5: Delayed weight-gradient protocol

A non-TE framework can participate in delayed wgrad capture by exposing a small
structural protocol on modules:

```python
class MyLinear(torch.nn.Linear):
    def need_backward_dw(self) -> bool:
        return True

    def backward_dw(self):
        return launch_or_compute_delayed_wgrad()
```

Captured callables receive a `backward_dw()` attribute. Framework schedulers can
call it at the same place they would call TE's delayed wgrad hook.

## Case 6: Optional TransformerEngine / FP8

Use one code path and enable FP8 only when TE and hardware support it.

```python
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe
    fp8_available, _ = te.is_fp8_available(return_reason=True)
except Exception:
    te = None
    fp8_available = False

kwargs = {}
if fp8_available:
    kwargs.update(enabled=True, recipe=recipe.DelayedScaling())

graphed = make_graphed_callables(module, sample_args, **kwargs)
```

Without TE, FP8/amax/cache-quantized options raise `RuntimeError`; generic graph
features continue to work.

## Failure modes to surface to users

- Shape or kwarg-key changes after capture are invalid.
- Hooks registered before capture are rejected; register framework hooks after
  graphing, wrap the hook behavior inside the module forward, or pass
  non-capturable capture-time behavior through `capture_time_hooks`.
- Modules must be all training or all inference during capture.
- TE FP8 parity depends on TransformerEngine internals compatible with the pinned
  upstream baseline in `UPSTREAM.md`.

## Recommended validation matrix

Run all three groups before publishing a framework integration:

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m pytest -q tests/test_cuda_runtime.py
PYTHONPATH=src python -m pytest -q tests/test_te_parity.py
```

The TE parity suite skips when TE/CUDA are absent and runs behavior comparisons
when they are present.
