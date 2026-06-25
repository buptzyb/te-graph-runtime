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
    assert hasattr(graphed, "reset")
    graphed.reset()


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
    with pytest.raises(TypeError, match="scale"):
        graphed(x)


def test_cuda_multiple_modules_with_tuple_kwargs_without_te(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_no_te(monkeypatch)

    class Scale(torch.nn.Module):
        def forward(self, x, scale):
            return x * scale

    modules = (Scale().cuda(), Scale().cuda())
    samples = (
        (torch.ones(2, 4, device="cuda", requires_grad=True),),
        (torch.ones(2, 4, device="cuda", requires_grad=True),),
    )
    kwargs = (
        {"scale": torch.ones(2, 4, device="cuda")},
        {"scale": torch.ones(2, 4, device="cuda") * 2},
    )
    graphed = graph.make_graphed_callables(
        modules, samples, sample_kwargs=kwargs, allow_unused_input=True
    )
    x = torch.ones(2, 4, device="cuda", requires_grad=True)
    assert torch.allclose(graphed[0](x, scale=torch.full_like(x, 3)), x * 3)
    assert torch.allclose(graphed[1](x, scale=torch.full_like(x, 4)), x * 4)


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


def test_cuda_order_buffer_reuse_without_te(monkeypatch: pytest.MonkeyPatch) -> None:
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
        _reuse_graph_input_output_buffers=True,
    )
    x = torch.randn(2, 4, device="cuda", requires_grad=True)
    forwards[0](x).sum().backward()
    assert layers[0].weight.grad is not None


def test_cuda_warmup_hooks_without_te(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_no_te(monkeypatch)
    calls = {"pre": 0, "post": 0}
    model = torch.nn.Linear(4, 4).cuda()
    sample = torch.randn(2, 4, device="cuda", requires_grad=True)
    graph.make_graphed_callables(
        model,
        (sample,),
        allow_unused_input=True,
        pre_warmup_hook=lambda: calls.__setitem__("pre", calls["pre"] + 1),
        post_warmup_hook=lambda: calls.__setitem__("post", calls["post"] + 1),
    )
    assert calls == {"pre": 1, "post": 1}


def test_cuda_eval_mode_falls_back_to_original_forward_without_te(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_no_te(monkeypatch)

    class ModeAware(torch.nn.Module):
        def forward(self, x):
            return x * (2 if self.training else 3)

    model = ModeAware().cuda().train()
    sample = torch.ones(2, 4, device="cuda", requires_grad=True)
    graphed = graph.make_graphed_callables(model, (sample,), allow_unused_input=True)
    graphed.eval()
    x = torch.ones(2, 4, device="cuda", requires_grad=True)
    assert torch.allclose(graphed(x), torch.full_like(x, 3))


def test_cuda_structural_backward_dw_protocol_without_te(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_no_te(monkeypatch)

    class DelayWgrad(torch.nn.Linear):
        def need_backward_dw(self):
            return True

        def backward_dw(self):
            # Real frameworks would launch delayed wgrad work here. This tensor op
            # keeps the CUDA graph capture non-empty without changing parameters.
            return self.weight.detach().sum() * 0

    model = DelayWgrad(4, 4).cuda()
    sample = torch.randn(2, 4, device="cuda", requires_grad=True)
    graphed = graph.make_graphed_callables(model, (sample,), allow_unused_input=True)
    assert hasattr(graphed, "backward_dw")
    graphed(torch.randn(2, 4, device="cuda", requires_grad=True)).sum().backward()
    graphed.backward_dw()
