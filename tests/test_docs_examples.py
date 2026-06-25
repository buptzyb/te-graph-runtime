import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_examples_compile() -> None:
    for path in sorted((ROOT / "examples").glob("*.py")):
        py_compile.compile(str(path), doraise=True)


def test_custom_framework_docs_cover_required_cases() -> None:
    text = (ROOT / "docs" / "custom_framework_integration.md").read_text()
    required = [
        "Plain module capture",
        "Framework kwargs",
        "Interleaved pipeline schedules",
        "Delayed weight-gradient protocol",
        "Optional TransformerEngine / FP8",
        "Recommended validation matrix",
    ]
    for phrase in required:
        assert phrase in text


def test_readme_points_to_examples_and_test_matrix() -> None:
    text = (ROOT / "README.md").read_text()
    for phrase in [
        "custom_framework_minimal.py",
        "custom_framework_pipeline.py",
        "optional_te_fp8.py",
        "tests/test_te_parity.py",
        "tests/test_cuda_runtime.py",
    ]:
        assert phrase in text
