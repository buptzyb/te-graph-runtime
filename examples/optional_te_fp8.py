"""Optional TransformerEngine/FP8 integration example.

Run with:
    PYTHONPATH=../src python examples/optional_te_fp8.py
"""

from __future__ import annotations

import torch

from te_graph_runtime import make_graphed_callables


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for CUDA graph capture")

    kwargs = {}
    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common import recipe

        fp8_available, reason = te.is_fp8_available(return_reason=True)
        module = te.Linear(8, 8, params_dtype=torch.float32, device="cuda")
        if fp8_available:
            kwargs.update(enabled=True, recipe=recipe.DelayedScaling())
        else:
            print(f"FP8 disabled: {reason}")
    except Exception as exc:
        print(f"TransformerEngine unavailable, using torch.nn.Linear: {exc}")
        module = torch.nn.Linear(8, 8).cuda()

    sample = torch.randn(4, 8, device="cuda", requires_grad=True)
    graphed = make_graphed_callables(module, (sample,), allow_unused_input=True, **kwargs)
    output = graphed(torch.randn(4, 8, device="cuda", requires_grad=True))
    output.sum().backward()
    print(output.shape)


if __name__ == "__main__":
    main()
