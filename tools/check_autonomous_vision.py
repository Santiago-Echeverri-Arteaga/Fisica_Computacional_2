"""Ejecuta el código real de un notebook autónomo en CPU y registra duraciones.

No sustituye una sesión de Colab: comprueba descargas, API, entrenamiento y evaluación
en el Python que invoca este script. Resultados fuera de los notebooks fuente.
"""
import argparse
import json
import linecache
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", choices=["01", "02"])
    parser.add_argument("--dataset", choices=["flores", "cifar10"], default="flores")
    parser.add_argument("--check-challenges", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["KERAS_HOME"] = str(ROOT / "curso_2026_2/.artifacts/keras-cache")
    import nbformat
    from IPython.display import display

    path = next((ROOT / "curso_2026_2/trabajo_autonomo_4h").glob(args.notebook + "_*.ipynb"))
    document = nbformat.read(path, as_version=4)
    nbformat.validate(document)
    output = ROOT / "curso_2026_2/.artifacts" / f"autonomo-{args.notebook}-{args.dataset}"
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)
    namespace = {"__name__": "__main__", "display": display}
    timings = []
    start = time.perf_counter()
    for i, cell in enumerate(document.cells):
        if cell.cell_type != "code":
            continue
        source = cell.source
        if args.dataset != "flores":
            source = source.replace('DATASET = "flores"', 'DATASET = "cifar10"')
        print(f"CELL {i}: START", flush=True)
        tick = time.perf_counter()
        filename = f"{path.name}:cell-{i}"
        linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
        exec(compile(source, filename, "exec"), namespace)
        elapsed = time.perf_counter() - tick
        timings.append({"cell": i, "seconds": elapsed})
        print(f"CELL {i}: OK {elapsed:.2f}s", flush=True)
    if args.check_challenges:
        if args.notebook == "01":
            source = next(c.source for c in document.cells
                          if c.cell_type == "code" and "FILTROS_INICIALES = 16" in c.source)
            original_count = namespace["cnn"].count_params()
            exec(compile(source.replace("FILTROS_INICIALES = 16", "FILTROS_INICIALES = 8"),
                         "challenge-cnn", "exec"), namespace)
            assert namespace["cnn"].count_params() < original_count
        else:
            # Comprueba todas las variantes sugeridas, incluido un paso de gradiente real.
            x, y = next(iter(namespace["train_ds"]))
            for filters, residual in [(8, False), (12, True), (12, False)]:
                model = namespace["crear_personalizada"](filters, residual)
                namespace["compilar"](model)
                assert model(x).shape == (len(x), namespace["K"])
                loss, accuracy = model.train_on_batch(x, y)
                assert namespace["np"].isfinite(loss)
                print("CHALLENGE OK", filters, residual, model.count_params(), flush=True)
    report = {"notebook": path.name, "dataset": args.dataset,
              "challenges_checked": args.check_challenges,
              "tensorflow": namespace["tf"].__version__, "gpu": namespace["GPU"],
              "seconds": time.perf_counter() - start, "cells": timings}
    (output / "execution.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("SUCCESS", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
