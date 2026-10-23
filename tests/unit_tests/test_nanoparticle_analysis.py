import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


APP_DIR = Path(__file__).resolve().parents[2] / "examples/06.BTO_nanoparticle"


@pytest.fixture(scope="module")
def analysis():
    spec = importlib.util.spec_from_file_location(
        "nanoparticle_analysis", APP_DIR / "02.analyze.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(APP_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _write_sampling_output(output_dir, *, dump_fields=True, sampling_steps=12):
    config = {
        "size": 5,
        "radius": 1.0,
        "center": None,
        "temperature": 250.0,
        "dt": 0.002,
        "sampling_steps": sampling_steps,
        "surface_shell_thickness_cells": 0.0,
        "surface_k2_shift": 0.0,
    }
    (output_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    mask = np.sum((np.indices((5, 5, 5)) - 2) ** 2, axis=0) <= 1
    if sampling_steps > 0:
        box_mean = 0.1 * mask.sum() / mask.size
        np.savetxt(
            output_dir / "sample_dipole_avg.log",
            [[2, box_mean, 0.0, 0.0], [10, -box_mean, 0.0, 0.0]],
            delimiter=",", header="Step, ux, uy, uz",
        )
        np.savetxt(
            output_dir / "sample_thermo.log", [[2, 100.0], [10, 100.0]],
            delimiter=",", header="Step, Volume",
        )
    if dump_fields:
        trajectory_dir = output_dir / "traj"
        trajectory_dir.mkdir()
        for step, amplitude in ((2, 0.1), (10, -0.1)):
            field = np.zeros((5, 5, 5, 3))
            field[..., 0][mask] = amplitude
            np.save(trajectory_dir / f"sample_dipole_dump_{step}.npy", field)


def test_sampling_analysis_needs_no_bulk_run(analysis, tmp_path, monkeypatch):
    _write_sampling_output(tmp_path)

    def unexpected_bulk_read(*args, **kwargs):
        raise AssertionError("No bulk reference was requested.")

    monkeypatch.setattr(analysis, "load_bulk_soft_mode", unexpected_bulk_read)
    analysis.main(["--output-dir", str(tmp_path)])

    results = json.loads((tmp_path / "polarization_chi.json").read_text())
    assert results["n_samples"] == 2
    assert results["n_particle"] == 7
    np.testing.assert_allclose(results["mean_polarization_C_per_m2"], 0.0)
    chi = np.asarray(results["chi"])
    assert np.isfinite(chi).all()
    assert chi[0, 0] > 0.0
    np.testing.assert_array_equal(chi[1:, :], 0.0)
    profile = json.loads((tmp_path / "radial_soft_mode_profile.json").read_text())
    assert profile["n_snapshots"] == 2
    assert profile["bins"][0]["n_cells"] == 7
    assert profile["bins"][0]["mean_abs_u_A"] == pytest.approx(0.1)
    assert (tmp_path / "polarization.png").stat().st_size > 0
    assert (tmp_path / "dipole_snapshots.png").stat().st_size > 0


@pytest.mark.parametrize("sampling_steps", [0, 12])
def test_analysis_without_dumps_reports_skipped_plots(
    analysis, tmp_path, capsys, sampling_steps
):
    _write_sampling_output(
        tmp_path, dump_fields=False, sampling_steps=sampling_steps
    )

    analysis.main(["--output-dir", str(tmp_path)])

    results = json.loads((tmp_path / "polarization_chi.json").read_text())
    assert results["n_samples"] == (None if sampling_steps == 0 else 2)
    assert not (tmp_path / "radial_soft_mode_profile.json").exists()
    assert not (tmp_path / "dipole_snapshots.png").exists()
    assert "radial profile and snapshot plot skipped" in capsys.readouterr().out


def test_explicit_bulk_reference(analysis, tmp_path):
    _write_sampling_output(tmp_path)
    reference = tmp_path / "bulk.csv"
    np.savetxt(
        reference, [[0, 0.03, 0.04, 0.0], [1, -0.03, -0.04, 0.0]],
        delimiter=",", header="Step, ux, uy, uz",
    )

    assert analysis.load_bulk_soft_mode(reference) == pytest.approx(0.05)
    analysis.main([
        "--output-dir", str(tmp_path), "--bulk-reference", str(reference)
    ])

    assert (tmp_path / "polarization_chi.json").is_file()
    assert (tmp_path / "dipole_snapshots.png").stat().st_size > 0


def test_missing_explicit_reference_preserves_numerical_results(analysis, tmp_path):
    _write_sampling_output(tmp_path)

    with pytest.raises(FileNotFoundError, match="missing.csv"):
        analysis.main([
            "--output-dir", str(tmp_path),
            "--bulk-reference", str(tmp_path / "missing.csv"),
        ])

    assert (tmp_path / "polarization_chi.json").is_file()


def test_invalid_snapshot_is_reported_after_saving_results(analysis, tmp_path):
    _write_sampling_output(tmp_path)
    np.save(tmp_path / "traj/sample_dipole_dump_2.npy", np.zeros((3, 3, 3, 3)))

    with pytest.raises(ValueError, match="has shape"):
        analysis.main(["--output-dir", str(tmp_path)])

    assert (tmp_path / "polarization_chi.json").is_file()


def test_configured_sampling_requires_its_logs(analysis, tmp_path):
    _write_sampling_output(tmp_path, dump_fields=False)
    (tmp_path / "sample_thermo.log").unlink()

    with pytest.raises(FileNotFoundError, match="Missing thermo log"):
        analysis.main(["--output-dir", str(tmp_path)])
