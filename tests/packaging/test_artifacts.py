import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _run(command, *, cwd, env):
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    return result.stdout


def _copy_clean_source(destination):
    destination.mkdir()
    shutil.copytree(
        ROOT / "openferro",
        destination / "openferro",
        ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "*.egg-info", ".ipynb_checkpoints",
            "archived", "build",
        ),
    )
    for filename in ("pyproject.toml", "setup.py", "README.md", "LICENSE"):
        shutil.copy2(ROOT / filename, destination / filename)


def _assert_artifact_contents(source, wheel, sdist):
    required = {
        path.relative_to(source).as_posix()
        for path in (source / "openferro").rglob("*.py")
    }
    with zipfile.ZipFile(wheel) as archive:
        wheel_files = set(archive.namelist())
    assert required <= wheel_files

    with tarfile.open(sdist, "r:gz") as archive:
        sdist_files = {
            "/".join(Path(name).parts[1:]) for name in archive.getnames()
        }
    assert required <= sdist_files


def _install_and_smoke_test(artifact, environment_dir, outside_dir, env):
    _run(
        [sys.executable, "-m", "venv", "--system-site-packages", environment_dir],
        cwd=outside_dir,
        env=env,
    )
    python = environment_dir / "bin" / "python"
    _run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            artifact,
        ],
        cwd=outside_dir,
        env=env,
    )
    smoke_test = """
import importlib
from pathlib import Path
import pkgutil
import sys

import openferro as of

package_root = Path(of.__file__).resolve().parent
assert Path(sys.prefix) in package_root.parents, package_root
for module in pkgutil.walk_packages(of.__path__, prefix="openferro."):
    importlib.import_module(module.name)
system = of.System(of.SimpleCubic3D(1, 1, 1))
field = system.add_field("x")
assert field.get_values().shape == (1, 1, 1, 1)
print(package_root)
"""
    _run([python, "-c", smoke_test], cwd=outside_dir, env=env)


@pytest.mark.packaging
def test_packaging_extra_covers_no_isolation_build_requirements():
    metadata = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    build_requirements = set(metadata["build-system"]["requires"])
    packaging_requirements = set(
        metadata["project"]["optional-dependencies"]["package"]
    )
    missing_requirements = build_requirements - packaging_requirements

    assert not missing_requirements, (
        "The package extra must provide no-isolation build requirements: "
        f"{sorted(missing_requirements)}"
    )


@pytest.mark.packaging
def test_clean_sdist_and_wheel_install_outside_checkout(tmp_path):
    source = tmp_path / "source"
    distribution = tmp_path / "dist"
    outside = tmp_path / "outside"
    _copy_clean_source(source)
    distribution.mkdir()
    outside.mkdir()
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"

    _run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--outdir",
            distribution,
        ],
        cwd=source,
        env=env,
    )
    wheel, = distribution.glob("*.whl")
    sdist, = distribution.glob("*.tar.gz")
    _assert_artifact_contents(source, wheel, sdist)

    _install_and_smoke_test(wheel, tmp_path / "wheel-env", outside, env)
    _install_and_smoke_test(sdist, tmp_path / "sdist-env", outside, env)
