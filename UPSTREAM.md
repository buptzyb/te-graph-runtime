# Upstream Baseline

This repository starts from TransformerEngine's PyTorch CUDA graph helper.

- Upstream repo: `https://github.com/NVIDIA/TransformerEngine`
- Baseline tag: `v2.16`
- Baseline commit: `4220403e831d29e93868f7793693ea83f6b8b05b`
- Source file: `transformer_engine/pytorch/graph.py`
- Local baseline copy:
  `src/te_graph_runtime/_upstream/transformer_engine_v2_16_graph.py`

## Carried upstream patches

- NVIDIA/TransformerEngine PR #2937 (`Fix CUDA graph parameter grad lifetime`):
  adds `_clone_param_grads_on_return` and clones returned parameter-gradient
  slots by default so graph replay does not expose CUDA graph static buffers to
  downstream autograd users. This repo carries the PR on top of the v2.16
  baseline because it is relevant to the standalone runtime behavior.

## Update workflow

1. Add a new upstream copy for the next TransformerEngine release tag.
2. Diff the previous and new upstream copies.
3. Classify changes as public API, generic torch graph behavior, or
   TransformerEngine-specific behavior.
4. Apply generic behavior to the torch-only backend and TE-specific behavior to
   the optional TE adapter.
5. Keep `te_graph_runtime.make_graphed_callables` signature compatible with the
   selected TransformerEngine release.

