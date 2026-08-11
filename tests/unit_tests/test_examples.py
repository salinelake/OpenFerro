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
]


@pytest.mark.slow
@pytest.mark.parametrize(
    "script, arguments, expected_files",
    EXAMPLES,
    ids=("BaTiO3", "bcc_Fe", "sc_Heisenberg", "PTO_STO", "BTO_nanoparticle"),
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
