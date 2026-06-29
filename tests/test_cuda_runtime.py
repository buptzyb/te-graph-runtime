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


def test_cuda_module_compile_is_stabilized_before_capture_without_te(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_no_te(monkeypatch)
    model = torch.nn.Linear(4, 4).cuda()
    model.compile(dynamic=False, options={"triton.cudagraphs": False})
    sample = torch.randn(2, 4, device="cuda", requires_grad=True)

    graphed = graph.make_graphed_callables(
        model,
        (sample,),
        allow_unused_input=True,
        num_warmup_iters=1,
    )

    x = torch.randn(2, 4, device="cuda", requires_grad=True)
    graphed(x).sum().backward()
    assert model.weight.grad is not None


def _assert_not_cuda_capturing() -> None:
    is_current_stream_capturing = getattr(torch.cuda, "is_current_stream_capturing", None)
    if is_current_stream_capturing is not None:
        assert not is_current_stream_capturing()


def test_cuda_capture_time_hooks_run_only_during_capture_without_te(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_no_te(monkeypatch)

    class ScaleLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, x, scale):
            return self.linear(x) * scale

    model = ScaleLinear().cuda()
    sample = torch.randn(2, 4, device="cuda", requires_grad=True)
    sample_scale = torch.ones(2, 4, device="cuda")
    calls = {"pre": 0, "pre_kwargs": 0, "fwd": 0, "fwd_kwargs": 0, "bwd_pre": 0, "bwd": 0}

    def pre_hook(module, args):
        _assert_not_cuda_capturing()
        assert module is model
        assert len(args) == 1
        calls["pre"] += 1

    def pre_kwargs_hook(module, args, kwargs):
        _assert_not_cuda_capturing()
        assert module is model
        assert "scale" in kwargs
        calls["pre_kwargs"] += 1

    def fwd_hook(module, args, output):
        _assert_not_cuda_capturing()
        assert module is model
        assert isinstance(output, torch.Tensor)
        calls["fwd"] += 1

    def fwd_kwargs_hook(module, args, kwargs, output):
        _assert_not_cuda_capturing()
        assert module is model
        assert "scale" in kwargs
        assert isinstance(output, torch.Tensor)
        calls["fwd_kwargs"] += 1

    def bwd_pre_hook(module, grad_output):
        _assert_not_cuda_capturing()
        assert module is model
        assert len(grad_output) == 1
        calls["bwd_pre"] += 1

    def bwd_hook(module, grad_input, grad_output):
        _assert_not_cuda_capturing()
        assert module is model
        assert len(grad_input) > 0
        assert len(grad_output) == 1
        calls["bwd"] += 1

    graphed = graph.make_graphed_callables(
        model,
        (sample,),
        sample_kwargs={"scale": sample_scale},
        allow_unused_input=True,
        num_warmup_iters=2,
        capture_time_hooks=[
            {
                "forward_pre_hooks": {1: pre_hook, 2: pre_kwargs_hook},
                "forward_pre_hooks_with_kwargs": {2: True},
                "forward_hooks": {3: fwd_hook, 4: fwd_kwargs_hook},
                "forward_hooks_with_kwargs": {4: True},
                "backward_pre_hooks": {5: bwd_pre_hook},
                "backward_hooks": {6: bwd_hook},
            }
        ],
    )
    expected_calls = {name: 3 for name in calls}
    assert calls == expected_calls

    x = torch.randn(2, 4, device="cuda", requires_grad=True)
    scale = torch.full((2, 4), 2.0, device="cuda")
    graphed(x, scale=scale).sum().backward()
    torch.cuda.synchronize()
    assert calls == expected_calls
    graphed.reset()


def test_cuda_capture_time_hooks_follow_order_without_te(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_no_te(monkeypatch)
    layers = torch.nn.ModuleList([torch.nn.Linear(4, 4).cuda(), torch.nn.Linear(4, 4).cuda()])
    sample_args = tuple(
        (torch.randn(2, 4, device="cuda", requires_grad=True),) for _ in range(4)
    )
    counts = [{"fwd": 0, "bwd": 0}, {"fwd": 0, "bwd": 0}]

    def make_fwd_hook(layer_idx):
        def hook(module, args):
            _assert_not_cuda_capturing()
            assert module is layers[layer_idx]
            counts[layer_idx]["fwd"] += 1

        return hook

    def make_bwd_hook(layer_idx):
        def hook(module, grad_input, grad_output):
            _assert_not_cuda_capturing()
            assert module is layers[layer_idx]
            counts[layer_idx]["bwd"] += 1

        return hook

    forwards = graph.make_graphed_callables(
        tuple(layers),
        sample_args,
        allow_unused_input=True,
        num_warmup_iters=1,
        _order=[1, 2, 1, 2, -2, -1, -2, -1],
        capture_time_hooks=[
            {"forward_pre_hooks": {1: make_fwd_hook(0)}, "backward_hooks": {2: make_bwd_hook(0)}},
            {"forward_pre_hooks": {1: make_fwd_hook(1)}, "backward_hooks": {2: make_bwd_hook(1)}},
        ],
    )
    assert counts == [{"fwd": 4, "bwd": 4}, {"fwd": 4, "bwd": 4}]
    x = torch.randn(2, 4, device="cuda", requires_grad=True)
    forwards[1](forwards[0](x)).sum().backward()
    torch.cuda.synchronize()
    assert counts == [{"fwd": 4, "bwd": 4}, {"fwd": 4, "bwd": 4}]
    for forward in forwards:
        forward.reset()


def test_cuda_capture_time_hooks_validate_shape_without_te(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_no_te(monkeypatch)
    model = torch.nn.Linear(4, 4).cuda()
    sample = torch.randn(2, 4, device="cuda", requires_grad=True)
    with pytest.raises(ValueError, match="capture_time_hooks"):
        graph.make_graphed_callables(
            model,
            (sample,),
            allow_unused_input=True,
            capture_time_hooks=[None, None],
        )


def test_cuda_capture_time_hook_return_value_is_rejected_without_te(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_no_te(monkeypatch)
    model = torch.nn.Linear(4, 4).cuda()
    sample = torch.randn(2, 4, device="cuda", requires_grad=True)

    def bad_hook(module, args):
        return args

    with pytest.raises(RuntimeError, match="forward_pre_hooks"):
        graph.make_graphed_callables(
            model,
            (sample,),
            allow_unused_input=True,
            capture_time_hooks=[{"forward_pre_hooks": {1: bad_hook}}],
        )


def test_cuda_registered_backward_pre_hooks_are_rejected_without_te(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_no_te(monkeypatch)
    model = torch.nn.Linear(4, 4).cuda()
    sample = torch.randn(2, 4, device="cuda", requires_grad=True)
    handle = model.register_full_backward_pre_hook(lambda module, grad_output: None)
    try:
        with pytest.raises(RuntimeError, match="capture_time_hooks"):
            graph.make_graphed_callables(model, (sample,), allow_unused_input=True)
    finally:
        handle.remove()


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



def _new_linear_for_grad_lifetime_test():
    model = torch.nn.Linear(4, 4, bias=False).cuda()
    sample = torch.randn(2, 4, device="cuda", requires_grad=False)
    return model, sample


def _run_weight_grad_step(model, x, grad):
    output = model(x)
    output.backward(grad)
    torch.cuda.synchronize()


def _save_hooked_grads(param):
    seen = []

    def save_grad(grad):
        seen.append(grad)
        return grad

    return seen, param.register_hook(save_grad)


def test_cuda_parameter_grads_are_owned_without_te(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_no_te(monkeypatch)
    torch.manual_seed(501)
    model, sample = _new_linear_for_grad_lifetime_test()
    graphed = graph.make_graphed_callables(model, (sample,), allow_unused_input=True)
    seen_grads, hook = _save_hooked_grads(model.weight)
    try:
        _run_weight_grad_step(
            graphed,
            torch.randn(2, 4, device="cuda"),
            torch.randn(2, 4, device="cuda"),
        )
        assert len(seen_grads) == 1
        first_grad = seen_grads[0]
        first_grad_ptr = first_grad.data_ptr()
        first_grad_snapshot = first_grad.clone()

        model.zero_grad(set_to_none=True)
        _run_weight_grad_step(
            graphed,
            torch.randn(2, 4, device="cuda"),
            torch.randn(2, 4, device="cuda"),
        )

        assert len(seen_grads) == 2
        assert first_grad.data_ptr() == first_grad_ptr
        assert seen_grads[1].data_ptr() != first_grad_ptr
        torch.testing.assert_close(first_grad, first_grad_snapshot, rtol=0, atol=0)
    finally:
        hook.remove()
        graphed.reset()


def test_cuda_parameter_grad_accumulation_without_te(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_no_te(monkeypatch)
    torch.manual_seed(502)
    model, sample = _new_linear_for_grad_lifetime_test()
    graphed = graph.make_graphed_callables(model, (sample,), allow_unused_input=True)
    x1 = torch.randn(2, 4, device="cuda")
    g1 = torch.randn(2, 4, device="cuda")
    x2 = torch.randn(2, 4, device="cuda")
    g2 = torch.randn(2, 4, device="cuda")
    expected = torch.einsum("bo,bi->oi", g1, x1) + torch.einsum("bo,bi->oi", g2, x2)
    try:
        model.zero_grad(set_to_none=True)
        _run_weight_grad_step(graphed, x1, g1)
        _run_weight_grad_step(graphed, x2, g2)
        torch.testing.assert_close(model.weight.grad, expected, rtol=1e-5, atol=1e-5)
    finally:
        graphed.reset()


def test_cuda_skipped_parameter_grad_alias_is_preserved_without_te(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_no_te(monkeypatch)
    torch.manual_seed(503)
    model, sample = _new_linear_for_grad_lifetime_test()
    model.weight.skip_backward_post_hook = True
    graphed = graph.make_graphed_callables(model, (sample,), allow_unused_input=True)
    seen_grads, hook = _save_hooked_grads(model.weight)
    try:
        _run_weight_grad_step(
            graphed,
            torch.randn(2, 4, device="cuda"),
            torch.randn(2, 4, device="cuda"),
        )
        first_grad_ptr = seen_grads[0].data_ptr()
        model.zero_grad(set_to_none=True)
        _run_weight_grad_step(
            graphed,
            torch.randn(2, 4, device="cuda"),
            torch.randn(2, 4, device="cuda"),
        )
        assert len(seen_grads) == 2
        assert seen_grads[1].data_ptr() == first_grad_ptr
    finally:
        hook.remove()
        graphed.reset()


def test_cuda_parameter_grad_clone_can_be_disabled_without_te(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_no_te(monkeypatch)
    torch.manual_seed(504)
    model, sample = _new_linear_for_grad_lifetime_test()
    graphed = graph.make_graphed_callables(
        model,
        (sample,),
        allow_unused_input=True,
        _clone_param_grads_on_return=False,
    )
    seen_grads, hook = _save_hooked_grads(model.weight)
    try:
        _run_weight_grad_step(
            graphed,
            torch.randn(2, 4, device="cuda"),
            torch.randn(2, 4, device="cuda"),
        )
        first_grad_ptr = seen_grads[0].data_ptr()
        model.zero_grad(set_to_none=True)
        _run_weight_grad_step(
            graphed,
            torch.randn(2, 4, device="cuda"),
            torch.randn(2, 4, device="cuda"),
        )
        assert len(seen_grads) == 2
        assert seen_grads[1].data_ptr() == first_grad_ptr
    finally:
        hook.remove()
        graphed.reset()


def test_cuda_parameter_grad_clone_policy_is_snapshotted_without_te(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_no_te(monkeypatch)
    torch.manual_seed(505)
    model, sample = _new_linear_for_grad_lifetime_test()
    graphed = graph.make_graphed_callables(model, (sample,), allow_unused_input=True)
    model.weight.skip_backward_post_hook = True
    seen_grads, hook = _save_hooked_grads(model.weight)
    try:
        _run_weight_grad_step(
            graphed,
            torch.randn(2, 4, device="cuda"),
            torch.randn(2, 4, device="cuda"),
        )
        first_grad = seen_grads[0]
        first_grad_ptr = first_grad.data_ptr()
        first_grad_snapshot = first_grad.clone()
        model.zero_grad(set_to_none=True)
        _run_weight_grad_step(
            graphed,
            torch.randn(2, 4, device="cuda"),
            torch.randn(2, 4, device="cuda"),
        )
        assert len(seen_grads) == 2
        assert seen_grads[1].data_ptr() != first_grad_ptr
        torch.testing.assert_close(first_grad, first_grad_snapshot, rtol=0, atol=0)
    finally:
        hook.remove()
        graphed.reset()
