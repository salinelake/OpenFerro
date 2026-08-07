import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = [
    (
        ROOT / "examples/01.BTO_Cooling/npt.py",
        ("optimization.log", "thermo_300K.log", "field_300K_avg.log"),
    ),
    (
        ROOT / "examples/02.bcc_Fe_Heating/nvt.py",
        ("thermo_10K.log", "spin_10K_avg.log"),
    ),
    (
        ROOT / "examples/03.sc_Ising_Heating/nvt.py",
        ("thermo_700K.log", "spin_700K_avg.log"),
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize(
    "script, expected_files",
    EXAMPLES,
    ids=("BaTiO3", "bcc_Fe", "sc_Heisenberg"),
)
def test_maintained_example_tiny_mode(script, expected_files, tmp_path):
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
            "--tiny",
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
