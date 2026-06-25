"""Interleaved pipeline schedule example for a custom framework.

Run with:
    PYTHONPATH=../src python examples/custom_framework_pipeline.py
"""

from __future__ import annotations

import torch

from te_graph_runtime import make_graphed_callables


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for CUDA graph capture")

    layers = torch.nn.ModuleList([torch.nn.Linear(8, 8).cuda(), torch.nn.Linear(8, 8).cuda()])
    order = [1, 2, 1, 2, -2, -1, -2, -1]
    sample_args = tuple(
        (torch.randn(4, 8, device="cuda", requires_grad=True),) for _ in range(4)
    )

    layer_forwards = make_graphed_callables(
        tuple(layers),
        sample_args,
        _order=order,
        allow_unused_input=True,
        _reuse_graph_input_output_buffers=True,
    )

    hidden_states = torch.randn(4, 8, device="cuda", requires_grad=True)
    output = layer_forwards[1](layer_forwards[0](hidden_states))
    output.sum().backward()
    print(len(layer_forwards), output.shape)


if __name__ == "__main__":
    main()
