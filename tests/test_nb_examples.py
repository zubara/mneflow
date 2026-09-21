"""Execute the example notebooks end-to-end.

This catches breakage from mne/tensorflow/mneflow API changes that unit
tests wouldn't see (wrong kwargs, moved classes, changed return shapes) by
actually running the notebooks users follow.

No real (multi-GB) datasets are downloaded here. Each notebook has a small
CI branch -- gated by the MNEFLOW_CI env var, in its own "parameters" cell
near the top -- that swaps in a lightweight synthetic dataset of the same
shape instead of downloading the MNE "sample"/"multimodal" data. Training
length is likewise controlled by MNEFLOW_N_EPOCHS. See the parameters cell
in each notebook for details.

Cells tagged "skip-ci" are dropped before execution: currently just the
source-space localization section of regression_example.ipynb, which needs
a real MNE forward solution that has no lightweight synthetic substitute.
"""
import os
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"

# Execution order matters: mneflow_save_restore.ipynb and
# own_graph_example.ipynb both reload the tfrecords (and, for
# save_restore, the trained model) that mneflow_example_tf2.ipynb writes
# to MNEFLOW_DATA_PATH, so it must run first.
NOTEBOOKS = [
    "mneflow_example_tf2.ipynb",
    "mneflow_save_restore.ipynb",
    "own_graph_example.ipynb",
    "regression_example.ipynb",
]

SKIP_CI_TAG = "skip-ci"


def _execute_notebook(nb_path):
    nb = nbformat.read(nb_path, as_version=4)
    nb.cells = [
        cell for cell in nb.cells
        if SKIP_CI_TAG not in cell.get("metadata", {}).get("tags", [])
    ]
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(EXAMPLES_DIR)}},
    )
    client.execute()


@pytest.mark.slow
def test_example_notebooks_run(tmp_path, monkeypatch):
    """Run every example notebook with synthetic data and a handful of
    training epochs; fail with the offending cell's traceback if any
    notebook raises."""
    monkeypatch.setenv("MNEFLOW_CI", "1")
    monkeypatch.setenv("MNEFLOW_N_EPOCHS", "3")
    monkeypatch.setenv("MNEFLOW_DATA_PATH", str(tmp_path) + os.sep)
    monkeypatch.setenv("MPLBACKEND", "Agg")

    for name in NOTEBOOKS:
        try:
            _execute_notebook(EXAMPLES_DIR / name)
        except CellExecutionError as exc:
            pytest.fail(f"{name} raised during execution:\n{exc}", pytrace=False)
