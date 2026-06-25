import inspect
import os
import subprocess
import sys

import pytest

from te_graph_runtime import (
    UPSTREAM_TE_COMMIT,
    UPSTREAM_TE_GRAPH_PATH,
    UPSTREAM_TE_VERSION,
    make_graphed_callables,
)
import te_graph_runtime.graph as graph


def test_public_import() -> None:
    assert callable(make_graphed_callables)
    assert UPSTREAM_TE_VERSION == "v2.16"
    assert UPSTREAM_TE_COMMIT == "4220403e831d29e93868f7793693ea83f6b8b05b"
    assert UPSTREAM_TE_GRAPH_PATH == "transformer_engine/pytorch/graph.py"


def test_package_import_is_lazy() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str((os.getcwd() + "/src")), env.get("PYTHONPATH", "")]
    )
    code = (
        "import sys; "
        "import te_graph_runtime; "
        "print('torch' in sys.modules); "
        "print('transformer_engine' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )
    assert result.stdout.splitlines() == ["False", "False"]


def test_no_make_graphed_callables_delegation() -> None:
    source = inspect.getsource(graph)
    assert "transformer_engine.pytorch.graph" not in source
    assert "torch.cuda.make_graphed_callables(" not in source
    assert "make_graphed_callables as" not in source


def test_fp8_options_require_te_when_te_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(graph, "_TE_AVAILABLE", False)
    monkeypatch.setattr(graph, "_TE_IMPORT_ERROR", ImportError("no te"))
    with pytest.raises(RuntimeError, match="requires transformer_engine"):
        graph.make_graphed_callables(object(), (), enabled=True)


def test_false_enabled_tuple_is_not_te_specific(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(graph, "_TE_AVAILABLE", False)
    monkeypatch.setattr(graph, "_TE_IMPORT_ERROR", ImportError("no te"))
    with pytest.raises(TypeError, match="Graphing"):
        graph.make_graphed_callables((object(), object()), ((), ()), enabled=(False, False))
