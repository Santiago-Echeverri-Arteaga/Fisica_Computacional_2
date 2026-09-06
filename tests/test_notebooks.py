"""Pruebas que no requieren ejecutar los modelos."""

from pathlib import Path

from tools.validate_notebooks import notebook_paths, notebook_tier, validate_structure


def test_complete_notebook_collection_exists() -> None:
    paths = {path.name for path in notebook_paths()}
    assert len(paths) == 29
    assert "00_protocolo_evaluacion.ipynb" in paths
    assert "11_knn_svm.ipynb" in paths
    assert "31_activaciones_diseno_y_comparacion.ipynb" in paths
    assert "71_transformers_y_llm.ipynb" in paths
    assert "80_plantilla_proyecto_final.ipynb" in paths


def test_notebooks_have_valid_structure_and_clean_outputs() -> None:
    for path in notebook_paths():
        validate_structure(Path(path))
        assert notebook_tier(path) in {"base", "tensorflow", "pytorch", "tensorflow-network"}
