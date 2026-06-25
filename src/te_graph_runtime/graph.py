"""Public TE-compatible CUDA graph API.

This bootstrap keeps the TransformerEngine v2.16 function signature stable while
the standalone torch-only backend is split out from the upstream implementation.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

_T = TypeVar("_T")
SingleOrTuple = Union[_T, Tuple[_T, ...]]

UPSTREAM_TE_VERSION = "v2.16"
UPSTREAM_TE_COMMIT = "4220403e831d29e93868f7793693ea83f6b8b05b"
UPSTREAM_TE_GRAPH_PATH = "transformer_engine/pytorch/graph.py"


def _load_te_make_graphed_callables() -> Optional[Callable[..., Any]]:
    try:
        from transformer_engine.pytorch.graph import (  # type: ignore[import-not-found]
            make_graphed_callables as te_make_graphed_callables,
        )
    except Exception:
        return None
    return te_make_graphed_callables


def _bool_option_enabled(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, tuple):
        return any(_bool_option_enabled(item) for item in value)
    return True


def _uses_te_only_options(
    *,
    fp8_enabled: Any,
    fp8_calibrating: Any,
    fp8_recipe: Any,
    fp8_group: Any,
    fp8_weight_caching: Any,
    enabled: Any,
    calibrating: Any,
    recipe: Any,
    amax_reduction_group: Any,
    cache_quantized_params: Any,
) -> bool:
    return any(
        (
            _bool_option_enabled(fp8_enabled),
            _bool_option_enabled(fp8_calibrating),
            fp8_recipe is not None,
            fp8_group is not None,
            _bool_option_enabled(fp8_weight_caching),
            _bool_option_enabled(enabled),
            _bool_option_enabled(calibrating),
            recipe is not None,
            amax_reduction_group is not None,
            _bool_option_enabled(cache_quantized_params),
        )
    )


def _uses_extended_torch_options(
    *,
    sample_kwargs: Any,
    _order: Any,
    _num_layers_per_chunk: Any,
    retain_graph_in_backward: bool,
    _reuse_graph_input_output_buffers: bool,
    pre_warmup_hook: Any,
    post_warmup_hook: Any,
) -> bool:
    return any(
        (
            sample_kwargs is not None,
            _order is not None,
            _num_layers_per_chunk is not None,
            retain_graph_in_backward,
            _reuse_graph_input_output_buffers,
            pre_warmup_hook is not None,
            post_warmup_hook is not None,
        )
    )


def make_graphed_callables(
    modules: SingleOrTuple[Callable],
    sample_args: SingleOrTuple[Tuple["torch.Tensor", ...]],
    num_warmup_iters: int = 3,
    allow_unused_input: bool = False,
    sample_kwargs: Optional[SingleOrTuple[Dict[str, Any]]] = None,
    fp8_enabled: Optional[SingleOrTuple[bool]] = None,
    fp8_calibrating: Optional[bool] = None,
    fp8_recipe: Optional[Any] = None,
    fp8_group: Optional[Any] = None,
    fp8_weight_caching: Optional[bool] = None,
    enabled: Optional[SingleOrTuple[bool]] = None,
    calibrating: Optional[bool] = None,
    recipe: Optional[Any] = None,
    amax_reduction_group: Optional[Any] = None,
    cache_quantized_params: Optional[bool] = None,
    _order: Optional[List[int]] = None,
    _num_layers_per_chunk: Optional[List[int]] = None,
    pool: Optional[Tuple[int, ...]] = None,
    retain_graph_in_backward: bool = False,
    _reuse_graph_input_output_buffers: bool = False,
    pre_warmup_hook: Optional[Callable] = None,
    post_warmup_hook: Optional[Callable] = None,
) -> Union[Callable, Tuple[Callable, ...]]:
    """Make CUDA graph versions of callables with a TE-compatible API.

    When TransformerEngine is installed, this delegates to the upstream TE
    implementation. Without TransformerEngine, the bootstrap supports the basic
    PyTorch ``torch.cuda.make_graphed_callables`` argument subset and reports a
    clear error for TE-only or not-yet-split extended behavior.
    """

    te_make_graphed_callables = _load_te_make_graphed_callables()
    if te_make_graphed_callables is not None:
        return te_make_graphed_callables(
            modules,
            sample_args,
            num_warmup_iters=num_warmup_iters,
            allow_unused_input=allow_unused_input,
            sample_kwargs=sample_kwargs,
            fp8_enabled=fp8_enabled,
            fp8_calibrating=fp8_calibrating,
            fp8_recipe=fp8_recipe,
            fp8_group=fp8_group,
            fp8_weight_caching=fp8_weight_caching,
            enabled=enabled,
            calibrating=calibrating,
            recipe=recipe,
            amax_reduction_group=amax_reduction_group,
            cache_quantized_params=cache_quantized_params,
            _order=_order,
            _num_layers_per_chunk=_num_layers_per_chunk,
            pool=pool,
            retain_graph_in_backward=retain_graph_in_backward,
            _reuse_graph_input_output_buffers=_reuse_graph_input_output_buffers,
            pre_warmup_hook=pre_warmup_hook,
            post_warmup_hook=post_warmup_hook,
        )

    if _uses_te_only_options(
        fp8_enabled=fp8_enabled,
        fp8_calibrating=fp8_calibrating,
        fp8_recipe=fp8_recipe,
        fp8_group=fp8_group,
        fp8_weight_caching=fp8_weight_caching,
        enabled=enabled,
        calibrating=calibrating,
        recipe=recipe,
        amax_reduction_group=amax_reduction_group,
        cache_quantized_params=cache_quantized_params,
    ):
        raise RuntimeError(
            "TransformerEngine-only graph options require transformer_engine. "
            "Install te-graph-runtime[te] or disable FP8/TE-specific options."
        )

    if _uses_extended_torch_options(
        sample_kwargs=sample_kwargs,
        _order=_order,
        _num_layers_per_chunk=_num_layers_per_chunk,
        retain_graph_in_backward=retain_graph_in_backward,
        _reuse_graph_input_output_buffers=_reuse_graph_input_output_buffers,
        pre_warmup_hook=pre_warmup_hook,
        post_warmup_hook=post_warmup_hook,
    ):
        raise NotImplementedError(
            "The torch-only extended backend is not implemented in this bootstrap. "
            "Install transformer_engine for the TE v2.16 behavior or use only the "
            "basic PyTorch make_graphed_callables argument subset."
        )

    import torch

    return torch.cuda.make_graphed_callables(
        modules,
        sample_args,
        num_warmup_iters=num_warmup_iters,
        allow_unused_input=allow_unused_input,
        pool=pool,
    )

