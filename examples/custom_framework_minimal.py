"""Minimal custom-framework adapter example.

Run with:
    PYTHONPATH=../src python examples/custom_framework_minimal.py
"""

from __future__ import annotations

import torch

from te_graph_runtime import make_graphed_callables


class FrameworkBlock(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor, route_scale: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states) * route_scale


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for CUDA graph capture")

    block = FrameworkBlock(hidden_size=8).cuda()
    sample = torch.randn(4, 8, device="cuda", requires_grad=True)
    sample_scale = torch.ones(4, 8, device="cuda")

    graphed_block = make_graphed_callables(
        block,
        (sample,),
        sample_kwargs={"route_scale": sample_scale},
        allow_unused_input=True,
    )

    hidden_states = torch.randn(4, 8, device="cuda", requires_grad=True)
    route_scale = torch.full((4, 8), 2.0, device="cuda")
    output = graphed_block(hidden_states, route_scale=route_scale)
    output.sum().backward()
    print(output.shape)


if __name__ == "__main__":
    main()
