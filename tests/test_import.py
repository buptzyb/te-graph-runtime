from te_graph_runtime import (
    UPSTREAM_TE_COMMIT,
    UPSTREAM_TE_GRAPH_PATH,
    UPSTREAM_TE_VERSION,
    make_graphed_callables,
)


def test_public_import() -> None:
    assert callable(make_graphed_callables)
    assert UPSTREAM_TE_VERSION == "v2.16"
    assert UPSTREAM_TE_COMMIT == "4220403e831d29e93868f7793693ea83f6b8b05b"
    assert UPSTREAM_TE_GRAPH_PATH == "transformer_engine/pytorch/graph.py"

