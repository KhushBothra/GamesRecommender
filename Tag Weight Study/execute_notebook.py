"""Execute the complete study notebook and verify CLI/notebook agreement.

Run with the study environment: python execute_notebook.py. A prior CLI run is
required, so comparison checks are against independently generated results.
"""

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

from jupyter_client import KernelManager
import nbformat
from nbclient import NotebookClient


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    notebook_path = root / "Empirical Tag Weight Decay.ipynb"
    input_paths = [root / "Tag Weights DB/steamspy_tag_data.csv",
                   root / "steamspy_tag_data_indexed.csv",
                   root / "steamspy_average_decay_by_n_tags.csv"]
    before_inputs = {str(p.relative_to(root)): digest(p) for p in input_paths}
    results_dir = root / "results"
    result_paths = [results_dir / name for name in (
        "model_scores.csv", "fitted_parameters.csv", "per_group_errors.csv", "predictions.csv",
        "empirical_targets.csv", "selected_model.json")]
    result_paths.extend(results_dir / "generalization" / name for name in (
        "model_scores.csv", "per_group_errors.csv", "predictions.csv", "split_assignments.csv",
        "split_targets.csv", "length_rates.csv", "leave_length_out_errors.csv", "training_models.json",
        "selected_evaluation_model.json", "selected_length_aware_model.json", "paired_bootstrap.json",
        "run_manifest.json"))
    if not all(p.exists() for p in result_paths):
        raise RuntimeError("Run fit_decay.py and generalized_decay.py first to create independent comparison results.")
    before_results = {str(p.relative_to(results_dir)): digest(p) for p in result_paths}
    notebook = nbformat.read(notebook_path, as_version=4)
    with tempfile.TemporaryDirectory(prefix="steam-tag-decay-jupyter-") as runtime:
        env = {"IPYTHONDIR": str(Path(runtime) / "ipython"),
               "JUPYTER_RUNTIME_DIR": str(Path(runtime) / "runtime"),
               "MPLCONFIGDIR": str(Path(runtime) / "matplotlib"),
               "PYDEVD_DISABLE_FILE_VALIDATION": "1"}
        with patch.dict(os.environ, env):
            manager = KernelManager(kernel_name="python3")
            # Use this exact interpreter without installing a user-wide kernel.
            manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
            client = NotebookClient(notebook, km=manager, timeout=180,
                                    resources={"metadata": {"path": str(root)}})
            client.execute()
    nbformat.validate(notebook)
    for relative, expected in before_inputs.items():
        if digest(root / relative) != expected:
            raise AssertionError(f"Notebook modified an empirical input: {relative}")
    for name, expected in before_results.items():
        if digest(results_dir / name) != expected:
            raise AssertionError(f"CLI and notebook results differ: {name}")
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    if any(c.execution_count is None for c in code_cells):
        raise AssertionError("Notebook has unexecuted code cells.")
    nbformat.write(notebook, notebook_path)
    validation = {"code_cells_executed": len(code_cells), "cell_errors": 0,
                  "all_empirical_inputs_preserved": True,
                  "cli_notebook_numerical_outputs_identical": True,
                  "compared_outputs": before_results}
    (results_dir / "notebook_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(f"Executed {len(code_cells)} code cells; no errors.")
    print(f"All three empirical inputs are unchanged; all {len(before_results)} numerical exports match the CLI byte for byte.")
    print(f"Saved: {notebook_path}")


if __name__ == "__main__":
    main()
