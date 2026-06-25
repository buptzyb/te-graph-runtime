import copy
import inspect

import pytest

torch = pytest.importorskip("torch")
te_graph = pytest.importorskip("transformer_engine.pytorch.graph")
te = pytest.importorskip("transformer_engine.pytorch")

from te_graph_runtime import graph as runtime_graph


def _signature_shape(fn):
    sig = inspect.signature(fn)
    return tuple((name, param.kind, param.default) for name, param in sig.parameters.items())


def test_signature_matches_installed_te() -> None:
    runtime_sig = inspect.signature(runtime_graph.make_graphed_callables)
    te_sig = inspect.signature(te_graph.make_graphed_callables)
    for name, te_param in te_sig.parameters.items():
        assert name in runtime_sig.parameters
        runtime_param = runtime_sig.parameters[name]
        assert runtime_param.kind == te_param.kind
        assert runtime_param.default == te_param.default

    extra_params = set(runtime_sig.parameters) - set(te_sig.parameters)
    # Older installed TE builds can be behind our v2.16+PR2937 surface. If the
    # installed TE already includes these parameters, they are compared above.
    assert extra_params <= {
        "pre_warmup_hook",
        "post_warmup_hook",
        "_clone_param_grads_on_return",
    }
    for name in extra_params:
        assert runtime_sig.parameters[name].default is None if name.endswith("_hook") else True


def test_deprecated_conflict_errors_match_te(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_graph, "_TE_AVAILABLE", None)
    ours_msg = None
    te_msg = None
    with pytest.raises(ValueError) as ours_exc:
        runtime_graph.make_graphed_callables(object(), (), fp8_enabled=True, enabled=True)
    ours_msg = str(ours_exc.value)
    with pytest.raises(ValueError) as te_exc:
        te_graph.make_graphed_callables(object(), (), fp8_enabled=True, enabled=True)
    te_msg = str(te_exc.value)
    assert ours_msg == te_msg


cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _clone_module(module: torch.nn.Module) -> torch.nn.Module:
    cloned = copy.deepcopy(module)
    cloned.load_state_dict(copy.deepcopy(module.state_dict()))
    return cloned.cuda()


def _grads(module: torch.nn.Module):
    return [None if p.grad is None else p.grad.detach().cpu().clone() for p in module.parameters()]


def _zero_grads(module: torch.nn.Module) -> None:
    for p in module.parameters():
        p.grad = None


def _run_step(module, x, kwargs=None):
    kwargs = kwargs or {}
    _zero_grads(module)
    x = x.detach().clone().requires_grad_(True)
    out = module(x, **kwargs)
    out.sum().backward()
    torch.cuda.synchronize()
    return out.detach().cpu(), x.grad.detach().cpu(), _grads(module)


def _assert_step_close(ref, got, *, atol=1e-6, rtol=1e-6):
    ref_out, ref_x_grad, ref_param_grads = ref
    got_out, got_x_grad, got_param_grads = got
    torch.testing.assert_close(got_out, ref_out, atol=atol, rtol=rtol)
    torch.testing.assert_close(got_x_grad, ref_x_grad, atol=atol, rtol=rtol)
    assert len(got_param_grads) == len(ref_param_grads)
    for got_grad, ref_grad in zip(got_param_grads, ref_param_grads):
        if ref_grad is None:
            assert got_grad is None
        else:
            torch.testing.assert_close(got_grad, ref_grad, atol=atol, rtol=rtol)


@cuda_required
def test_torch_module_basic_graph_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_graph, "_TE_AVAILABLE", None)
    torch.manual_seed(7)
    base = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.GELU(),
        torch.nn.Linear(16, 8),
    ).cuda()
    ref = _clone_module(base)
    got = _clone_module(base)
    sample_ref = torch.randn(4, 8, device="cuda", requires_grad=True)
    sample_got = sample_ref.detach().clone().requires_grad_(True)

    ref = te_graph.make_graphed_callables(ref, (sample_ref,), allow_unused_input=True)
    got = runtime_graph.make_graphed_callables(got, (sample_got,), allow_unused_input=True)

    x = torch.randn(4, 8, device="cuda", requires_grad=True)
    _assert_step_close(_run_step(ref, x), _run_step(got, x))


@cuda_required
def test_sample_kwargs_graph_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_graph, "_TE_AVAILABLE", None)

    class KwargModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(8, 8)

        def forward(self, x, scale, bias):
            return self.linear(x) * scale + bias

    torch.manual_seed(11)
    base = KwargModule().cuda()
    ref = _clone_module(base)
    got = _clone_module(base)
    sample = torch.randn(3, 8, device="cuda", requires_grad=True)
    sample_kwargs = {
        "scale": torch.ones(3, 8, device="cuda"),
        "bias": torch.zeros(3, 8, device="cuda"),
    }

    ref = te_graph.make_graphed_callables(
        ref,
        (sample,),
        sample_kwargs={k: v.clone() for k, v in sample_kwargs.items()},
        allow_unused_input=True,
    )
    got = runtime_graph.make_graphed_callables(
        got,
        (sample.detach().clone().requires_grad_(True),),
        sample_kwargs={k: v.clone() for k, v in sample_kwargs.items()},
        allow_unused_input=True,
    )

    x = torch.randn(3, 8, device="cuda", requires_grad=True)
    kwargs = {
        "scale": torch.full((3, 8), 2.0, device="cuda"),
        "bias": torch.full((3, 8), -0.5, device="cuda"),
    }
    _assert_step_close(_run_step(ref, x, kwargs), _run_step(got, x, kwargs))


@cuda_required
def test_ordered_pipeline_graph_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_graph, "_TE_AVAILABLE", None)
    torch.manual_seed(13)
    base = torch.nn.ModuleList([torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)]).cuda()
    ref_layers = _clone_module(base)
    got_layers = _clone_module(base)
    order = [1, 2, 1, 2, -2, -1, -2, -1]
    sample_ref = tuple(
        (torch.randn(3, 8, device="cuda", requires_grad=True),) for _ in range(4)
    )
    sample_got = tuple((args[0].detach().clone().requires_grad_(True),) for args in sample_ref)

    ref_forwards = te_graph.make_graphed_callables(
        tuple(ref_layers), sample_ref, allow_unused_input=True, _order=order
    )
    got_forwards = runtime_graph.make_graphed_callables(
        tuple(got_layers), sample_got, allow_unused_input=True, _order=order
    )

    replay_x = torch.randn(3, 8, device="cuda", requires_grad=True)

    def run(forwards, layers, x):
        _zero_grads(layers)
        x = x.detach().clone().requires_grad_(True)
        out = forwards[1](forwards[0](x))
        out.sum().backward()
        torch.cuda.synchronize()
        return out.detach().cpu(), x.grad.detach().cpu(), _grads(layers)

    _assert_step_close(
        run(ref_forwards, ref_layers, replay_x),
        run(got_forwards, got_layers, replay_x),
    )


@cuda_required
def test_te_linear_no_fp8_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_graph, "_TE_AVAILABLE", None)
    torch.manual_seed(17)
    base = te.Linear(8, 8, params_dtype=torch.float32, device="cuda")
    ref = _clone_module(base)
    got = _clone_module(base)
    sample = torch.randn(4, 8, device="cuda", requires_grad=True)

    ref = te_graph.make_graphed_callables(ref, (sample,), allow_unused_input=True)
    got = runtime_graph.make_graphed_callables(
        got, (sample.detach().clone().requires_grad_(True),), allow_unused_input=True
    )
    x = torch.randn(4, 8, device="cuda", requires_grad=True)
    _assert_step_close(_run_step(ref, x), _run_step(got, x))


@cuda_required
def test_te_linear_fp8_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    fp8_available, reason = te.is_fp8_available(return_reason=True)
    if not fp8_available:
        pytest.skip(reason)
    recipe_mod = pytest.importorskip("transformer_engine.common.recipe")
    monkeypatch.setattr(runtime_graph, "_TE_AVAILABLE", None)
    torch.manual_seed(19)
    base = te.Linear(16, 16, params_dtype=torch.float32, device="cuda")
    ref = _clone_module(base)
    got = _clone_module(base)
    sample = torch.randn(8, 16, device="cuda", requires_grad=True)
    fp8_recipe_ref = recipe_mod.DelayedScaling()
    fp8_recipe_got = recipe_mod.DelayedScaling()

    try:
        ref = te_graph.make_graphed_callables(
            ref,
            (sample,),
            allow_unused_input=True,
            enabled=True,
            recipe=fp8_recipe_ref,
        )
    except RuntimeError as exc:
        pytest.skip(f"installed TE FP8 graph capture failed: {exc}")
    got = runtime_graph.make_graphed_callables(
        got,
        (sample.detach().clone().requires_grad_(True),),
        allow_unused_input=True,
        enabled=True,
        recipe=fp8_recipe_got,
    )
    x = torch.randn(8, 16, device="cuda", requires_grad=True)
    _assert_step_close(_run_step(ref, x), _run_step(got, x), atol=1e-2, rtol=1e-2)
