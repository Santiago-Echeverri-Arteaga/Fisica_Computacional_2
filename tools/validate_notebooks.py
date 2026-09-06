"""Validaciones rápidas y ejecución opcional para notebooks del curso 2026-2."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COURSE = ROOT / "curso_2026_2"


def notebook_paths() -> list[Path]:
    return sorted(COURSE.rglob("*.ipynb"))


def validate_structure(path: Path) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("nbformat") != 4:
        raise ValueError(f"{path}: se esperaba nbformat 4")
    if not document.get("cells"):
        raise ValueError(f"{path}: notebook vacío")

    ids: set[str] = set()
    for index, cell in enumerate(document["cells"], start=1):
        cell_id = cell.get("id")
        if not cell_id or cell_id in ids:
            raise ValueError(f"{path}: id ausente o repetido en celda {index}")
        ids.add(cell_id)

        if cell["cell_type"] == "code":
            if cell.get("execution_count") is not None or cell.get("outputs"):
                raise ValueError(f"{path}: la celda {index} conserva salidas")
            source = "".join(cell.get("source", []))
            ast.parse(source, filename=f"{path}:celda-{index}")


def notebook_tier(path: Path) -> str:
    document = json.loads(path.read_text(encoding="utf-8"))
    return document.get("metadata", {}).get("fc2", {}).get("tier", "base")


def execute_notebook(path: Path, timeout: int) -> None:
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError as error:
        raise RuntimeError(
            "Para ejecutar instale: python -m pip install -r requirements-2026.txt"
        ) from error

    with path.open(encoding="utf-8") as stream:
        notebook = nbformat.read(stream, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="ejecutar notebooks de los tiers seleccionados")
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=["base"],
        help="tiers a ejecutar: base, tensorflow, pytorch, tensorflow-network o all",
    )
    parser.add_argument("--timeout", type=int, default=600, help="segundos máximos por celda")
    args = parser.parse_args()

    paths = notebook_paths()
    if not paths:
        raise SystemExit("No se encontraron notebooks en curso_2026_2")

    executed = 0
    for path in paths:
        validate_structure(path)
        tier = notebook_tier(path)
        should_execute = args.execute and ("all" in args.tiers or tier in args.tiers)
        if should_execute:
            execute_notebook(path, timeout=args.timeout)
            executed += 1
        print(f"OK  [{tier}] {path.relative_to(ROOT)}")

    print(f"{len(paths)} notebooks: estructura y sintaxis correctas")
    if args.execute:
        print(f"{executed} notebooks ejecutados para tiers: {', '.join(args.tiers)}")


if __name__ == "__main__":
    main()
