import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = [
    (
        ROOT / "examples/01.BTO_Cooling/npt.py",
        ("--tiny",),
        ("optimization.log", "thermo_300K.log", "field_300K_avg.log"),
    ),
    (
        ROOT / "examples/02.bcc_Fe_Heating/nvt.py",
        ("--tiny",),
        ("thermo_10K.log", "spin_10K_avg.log"),
    ),
    (
        ROOT / "examples/03.sc_Ising_Heating/nvt.py",
        ("--tiny",),
        ("thermo_700K.log", "spin_700K_avg.log"),
    ),
    (
        ROOT / "examples/04.PTOSTO_superlattice/npt.py",
        (
            "--lateral-size",
            "4",
            "--pto-layers",
            "2",
            "--sto-layers",
            "2",
            "--relax-time-ps",
            "0.002",
            "--drive-time-ps",
            "0.002",
            "--log-interval",
            "1",
            "--dump-interval",
            "1",
        ),
        (
            "relax.log",
            "relax_field_dump_0.npy",
            "drive.log",
            "drive_field_dump_0.npy",
        ),
    ),
    (
        ROOT / "examples/06.BTO_nanoparticle/01.npt.py",
        (
            "--size",
            "5",
            "--radius",
            "1.0",
            "--cooldown_stages",
            "2",
            "--cooldown-steps",
            "2",
            "--sampling-steps",
            "2",
            "--cooldown-log-interval",
            "1",
            "--sample-log-interval",
            "1",
            "--sample-dump-interval",
            "1",
            "--surface-shell-thickness-cells",
            "0",
            "--surface-k2-shift",
            "0",
        ),
        (
            "config.json",
            "cooldown_T320.0K_thermo.log",
            "cooldown_T250.0K_thermo.log",
            "cooldown_T250.0K_dipole_avg.log",
            "sample_thermo.log",
            "sample_dipole_avg.log",
            "traj/sample_dipole_dump_0.npy",
            "traj/sample_dipole_dump_1.npy",
        ),
    ),
    (
        ROOT / "examples/07.MetaDynamics/run.py",
        ("--quick",),
        ("HILLS", "run_summary.json"),
    ),
    (
        ROOT / "examples/08.BTO_MetaD/run.py",
        (),
        (
            "model.json",
            "settings.json",
            "metadynamics_thermo.log",
            "metadynamics_dipole_avg.log",
            "HILLS",
        ),
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize(
    "script, arguments, expected_files",
    EXAMPLES,
    ids=(
        "BaTiO3",
        "bcc_Fe",
        "sc_Heisenberg",
        "PTO_STO",
        "BTO_nanoparticle",
        "MetaDynamics",
        "BTO_MetaDynamics",
    ),
)
def test_maintained_example_smoke_mode(
    script, arguments, expected_files, tmp_path
):
    output_dir = tmp_path / "output"
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(ROOT), environment.get("PYTHONPATH")))
    )
    arguments = list(arguments)
    if script.parent.name == "08.BTO_MetaD":
        settings = json.loads((script.parent / "settings.json").read_text())
        settings["lattice_size"] = 2
        settings["seed"] = 17
        settings["metadynamics"].update(
            steps=2, pace=1, log_interval=1, grid_bin=[32, 32, 32]
        )
        settings_path = tmp_path / "smoke_settings.json"
        settings_path.write_text(json.dumps(settings), encoding="utf-8")
        arguments.extend(["--settings", str(settings_path)])
    else:
        arguments.extend(["--seed", "17"])

    result = subprocess.run(
        [
            sys.executable,
            script.name,
            *arguments,
            "--output-dir",
            str(output_dir),
        ],
        cwd=script.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (output_dir / "simulation.log").is_file()
    for filename in expected_files:
        output = output_dir / filename
        assert output.is_file(), f"missing {output}"
        assert output.stat().st_size > 0, f"empty {output}"
    if script.parent.name == "06.BTO_nanoparticle":
        with (output_dir / "config.json").open(encoding="utf-8") as stream:
            config = json.load(stream)
        assert config["size"] == 5
        assert config["radius"] == 1.0
        assert config["seed"] == 17
        assert config["output_dir"] == str(output_dir)
        first_dump = np.load(output_dir / "traj/sample_dipole_dump_0.npy")
        assert first_dump.shape == (5, 5, 5, 3)
        assert np.isfinite(first_dump).all()
        analysis_result = subprocess.run(
            [sys.executable, "02.analyze.py", "--output-dir", str(output_dir)],
            cwd=script.parent,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert analysis_result.returncode == 0, (
            analysis_result.stdout + analysis_result.stderr
        )
        for filename in (
            "polarization_chi.json", "polarization.png",
            "radial_soft_mode_profile.json", "dipole_snapshots.png",
        ):
            assert (output_dir / filename).stat().st_size > 0
        results = json.loads((output_dir / "polarization_chi.json").read_text())
        assert results["n_samples"] == 2
        assert results["n_particle"] == config["n_particle"]
        assert np.isfinite(results["chi"]).all()
    if script.parent.name == "08.BTO_MetaD":
        saved_settings = json.loads((output_dir / "settings.json").read_text())
        model = json.loads((output_dir / "model.json").read_text())
        assert saved_settings == settings
        hills = np.loadtxt(output_dir / "HILLS", comments="#", ndmin=2)
        thermo = np.loadtxt(
            output_dir / "metadynamics_thermo.log", comments="#", delimiter=",",
            ndmin=2,
        )
        dipole = np.loadtxt(
            output_dir / "metadynamics_dipole_avg.log", comments="#",
            delimiter=",", ndmin=2,
        )
        np.testing.assert_array_equal(hills[:, 0], [1, 2])
        np.testing.assert_array_equal(thermo[:, 0], dipole[:, 0])
        # Columns after Step are the six Voigt strains, then the volume.
        volumes = thermo[:, 7]
        assert np.all(volumes > 0.0)
        polarization_scale = (
            model["parameters"]["born"]["Z_star"]
            * 16.02176634
            * settings["lattice_size"] ** 3
            / volumes
        )
        polarization = dipole[:, 1:4] * polarization_scale[:, None]
        expected_q = np.column_stack(
            (
                polarization[:, 0] - polarization[:, 1],
                polarization[:, 1] - polarization[:, 2],
                polarization[:, 2],
            )
        )
        np.testing.assert_allclose(
            hills[:, 1:4], expected_q, rtol=2e-5, atol=1e-7
        )
