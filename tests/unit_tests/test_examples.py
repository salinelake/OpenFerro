import json
import os
from pathlib import Path
import subprocess
import sys

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
        ROOT / "examples/06.BTO_Nanoparticle/01.npt.py",
        (
            "--size",
            "5",
            "--radius",
            "1.0",
            "--minimization-steps",
            "2",
            "--equilibration-steps",
            "2",
            "--sampling-steps",
            "2",
            "--relax-log-interval",
            "1",
            "--sample-log-interval",
            "1",
            "--dump-field",
        ),
        (
            "config.json",
            "optimization.log",
            "relax_thermo.log",
            "traj/relax_dipole_avg.log",
            "sample_thermo.log",
            "sample_dipole_avg.log",
        ),
    ),
    (
        ROOT / "examples/07.MetaDynamics/run.py",
        ("--quick",),
        ("HILLS", "run_summary.json"),
    ),
    (
        ROOT / "examples/08.BTO_MetaD/run.py",
        ("--quick",),
        (
            "optimization.log",
            "equilibration_thermo.log",
            "metadynamics_thermo.log",
            "metadynamics_dipole_avg.log",
            "HILLS",
            "run_summary.json",
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

    result = subprocess.run(
        [
            sys.executable,
            script.name,
            *arguments,
            "--output-dir",
            str(output_dir),
            "--seed",
            "17",
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
    if script.parent.name == "06.BTO_Nanoparticle":
        with (output_dir / "config.json").open(encoding="utf-8") as stream:
            config = json.load(stream)
        assert config["size"] == 5
        assert config["radius"] == 1.0
        assert config["seed"] == 17
        assert config["output_dir"] == str(output_dir)
    if script.parent.name == "08.BTO_MetaD":
        with (output_dir / "run_summary.json").open(encoding="utf-8") as stream:
            summary = json.load(stream)
        px, py, pz = summary["final_polarization_C_per_m2"]
        q1, q2, q3 = summary["final_q_C_per_m2"]
        assert q1 == pytest.approx(px - py)
        assert q2 == pytest.approx(py - pz)
        assert q3 == pytest.approx(pz)
        assert summary["ensemble"] == "NPT"
        assert summary["pressure_bar"] == pytest.approx(-48000.0)
        assert summary["initial_volume_per_cell_angstrom3"] > 0.0
        assert summary["final_volume_per_cell_angstrom3"] > 0.0
        assert len(summary["final_global_strain"]) == 6
        polarization_scale = (
            summary["born_effective_charge"]
            * 16.02176634
            / summary["final_volume_per_cell_angstrom3"]
        )
        assert (px, py, pz) == pytest.approx(
            [
                polarization_scale * value
                for value in summary["final_mean_local_mode_angstrom"]
            ]
        )
        assert summary["epsilon_C_per_m2"] > 0.0
