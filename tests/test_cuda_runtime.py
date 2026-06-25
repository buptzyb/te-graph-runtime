import pytest

torch = pytest.importorskip("torch")

import te_graph_runtime.graph as graph

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _force_no_te(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(graph, "_TE_AVAILABLE", False)
    monkeypatch.setattr(graph, "_TE_IMPORT_ERROR", ImportError("forced no te"))


def test_cuda_basic_module_replay_without_te(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_no_te(monkeypatch)
    torch.manual_seed(123)
    model = torch.nn.Linear(4, 4).cuda()
    sample = torch.randn(2, 4, device="cuda", requires_grad=True)
    graphed = graph.make_graphed_callables(model, (sample,), allow_unused_input=True)

    x = torch.randn(2, 4, device="cuda", requires_grad=True)
    out = graphed(x)
    out.sum().backward()
    assert out.shape == (2, 4)
    assert model.weight.grad is not None


def test_cuda_sample_kwargs_without_te(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_no_te(monkeypatch)

    class Scale(torch.nn.Module):
        def forward(self, x, scale):
            return x * scale

    model = Scale().cuda()
    sample = torch.ones(2, 4, device="cuda", requires_grad=True)
    sample_scale = torch.ones(2, 4, device="cuda")
    graphed = graph.make_graphed_callables(
        model,
        (sample,),
        sample_kwargs={"scale": sample_scale},
        allow_unused_input=True,
    )

    x = torch.full((2, 4), 2.0, device="cuda", requires_grad=True)
    scale = torch.full((2, 4), 3.0, device="cuda")
    out = graphed(x, scale=scale)
    out.sum().backward()
    assert torch.allclose(out, torch.full((2, 4), 6.0, device="cuda"))


def test_cuda_order_without_te(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_no_te(monkeypatch)
    layers = torch.nn.ModuleList([torch.nn.Linear(4, 4).cuda(), torch.nn.Linear(4, 4).cuda()])
    sample_args = tuple(
        (torch.randn(2, 4, device="cuda", requires_grad=True),) for _ in range(4)
    )
    forwards = graph.make_graphed_callables(
        tuple(layers),
        sample_args,
        allow_unused_input=True,
        _order=[1, 2, 1, 2, -2, -1, -2, -1],
    )

    x = torch.randn(2, 4, device="cuda", requires_grad=True)
    out = forwards[0](x)
    out.sum().backward()
    assert len(forwards) == 4
    assert layers[0].weight.grad is not None
