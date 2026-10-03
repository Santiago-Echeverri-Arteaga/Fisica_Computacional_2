"""Pruebas que no requieren ejecutar los modelos."""

from pathlib import Path

from tools.validate_notebooks import notebook_paths, notebook_tier, validate_structure


def test_execution_artifacts_are_not_source_notebooks(tmp_path, monkeypatch) -> None:
    from tools import validate_notebooks

    source = tmp_path / "03_redes_fundamentos" / "30_clase.ipynb"
    artifact = tmp_path / ".artifacts" / "30_clase.ipynb"
    checkpoint = tmp_path / "03_redes_fundamentos" / ".ipynb_checkpoints" / "30_clase.ipynb"
    for path in (source, artifact, checkpoint):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(validate_notebooks, "COURSE", tmp_path)
    assert notebook_paths() == [source]


def test_complete_notebook_collection_exists() -> None:
    paths = {path.name for path in notebook_paths()}
    assert len(paths) == 27
    assert "36_taller_120_minutos.ipynb" in paths
    assert "00_protocolo_evaluacion.ipynb" in paths
    assert "11_knn_svm.ipynb" in paths
    assert "30_neurona_y_feedforward_desde_cero.ipynb" in paths
    assert not any(name.startswith(("31_", "34_", "35_")) for name in paths)
    assert "71_transformers_y_llm.ipynb" in paths
    assert "80_plantilla_proyecto_final.ipynb" in paths


def test_notebooks_have_valid_structure_and_clean_outputs() -> None:
    for path in notebook_paths():
        validate_structure(Path(path))
        assert notebook_tier(path) in {"base", "tensorflow", "pytorch", "tensorflow-network",
                                       "pytorch-network", "tensorflow-pytorch-network"}
