"""Keep text-file IO portable to Windows non-UTF-8 locales."""
import ast
from pathlib import Path


def test_path_text_io_declares_encoding():
    root = Path(__file__).resolve().parents[1]
    missing = []
    for folder in ("daylight_workbench", "tests", "qa"):
        for path in (root / folder).rglob("*.py"):
            if "evidence" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                if node.func.attr not in {"read_text", "write_text"}:
                    continue
                if not any(keyword.arg == "encoding" for keyword in node.keywords):
                    missing.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not missing, "Declare text IO encoding explicitly: " + ", ".join(missing)
