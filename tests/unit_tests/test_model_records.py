import importlib.util
import json
import math
from collections.abc import Mapping
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import openferro as of
from openferro.engine.ferroelectric import (
    get_short_range_3rdnn_isotropic,
    self_energy_onsite_isotropic,
    short_range_1stnn_isotropic,
    short_range_2ednn_isotropic,
)
from openferro.engine.magnetic import get_isotropic_exchange_energy_engine
from openferro.units import Constants


ROOT = Path(__file__).resolve().parents[2]
FERROELECTRIC_RECORDS = sorted((ROOT / "model_configs").glob("*.json")) + [
    ROOT / "examples/01.BTO_Cooling/BaTiO3.json"
]
MAGNETIC_RECORDS = [
    ROOT / "examples/02.bcc_Fe_Heating/bcc_Fe.json",
    ROOT / "examples/03.sc_Ising_Heating/sc_Heisenberg.json",
]
MAGNETIC_EXAMPLES = [
    (MAGNETIC_RECORDS[0], ROOT / "examples/02.bcc_Fe_Heating/nvt.py"),
    (MAGNETIC_RECORDS[1], ROOT / "examples/03.sc_Ising_Heating/nvt.py"),
]
ALL_RECORDS = FERROELECTRIC_RECORDS + MAGNETIC_RECORDS


def _load_record(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_example_module(path):
    spec = importlib.util.spec_from_file_location(f"record_test_{path.parent.name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_finite_numbers(value):
    if isinstance(value, Mapping):
        for item in value.values():
            _assert_finite_numbers(item)
    elif isinstance(value, list):
        for item in value:
            _assert_finite_numbers(item)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        assert math.isfinite(value)


def _ferroelectric_parameter_arrays(record, dtype):
    parameters = record["parameters"]
    onsite = parameters["onsite"]
    short = parameters["short_range"]
    elastic = parameters["elastic"]
    elastic_dipole = parameters["elastic_dipole"]
    born = parameters["born"]
    return {
        "onsite": jnp.asarray(
            (onsite["k2"], onsite["alpha"], onsite["gamma"]), dtype=dtype
        ),
        "short_range_1": jnp.asarray((short["j1"], short["j2"]), dtype=dtype),
        "short_range_2": jnp.asarray(
            (short["j3"], short["j4"], short["j5"]), dtype=dtype
        ),
        "short_range_3": jnp.asarray((short["j6"], short["j7"]), dtype=dtype),
        "elastic": jnp.asarray(
            (elastic["B11"], elastic["B12"], elastic["B44"]), dtype=dtype
        ),
        "elastic_dipole": jnp.asarray(
            (
                elastic_dipole["B1xx"],
                elastic_dipole["B1yy"],
                elastic_dipole["B4yz"],
            ),
            dtype=dtype,
        ),
        "ewald_prefactor": jnp.asarray(
            (born["Z_star"] ** 2 / born["epsilon_inf"],), dtype=dtype
        ),
    }


def _magnetic_exchange_couplings(record, dtype):
    parameters = record["parameters"]
    units = record["units"]
    conventions = record["conventions"]
    unit_to_ev = {"mRy": Constants.mRy, "J": Constants.Joule}[
        units["exchange_source"]
    ]
    pair_factor = 2.0 if conventions["source_pair_counting"] == "ordered" else 1.0
    return (
        jnp.asarray(parameters["exchange_source_values"], dtype=dtype)
        * unit_to_ev
        * pair_factor
        / parameters["moment_mu_B"] ** 2
    )


@pytest.mark.parametrize("path", ALL_RECORDS, ids=lambda path: path.stem)
def test_model_record_carries_provenance_and_finite_data(path):
    record = _load_record(path)

    assert record["schema_version"] == 1
    assert record["model"]["id"]
    assert record["model"]["material"]
    assert record["model"]["method"]
    assert record["model"]["kind"]
    assert record["model"]["citation"]["text"]
    assert record["model"]["citation"]["doi"].startswith("10.")
    assert isinstance(record["lattice"], dict)
    assert isinstance(record["units"], dict)
    assert isinstance(record["conventions"], dict)
    assert isinstance(record["parameters"], dict)

    reference = record["reference_observable"]
    assert reference["name"]
    assert reference["units"]
    assert reference["atol"] > 0.0
    _assert_finite_numbers(record)


@pytest.mark.scientific
@pytest.mark.parametrize(
    "path", FERROELECTRIC_RECORDS, ids=lambda path: path.stem
)
def test_ferroelectric_record_matches_production_engines(path):
    record = _load_record(path)
    assert record["model"]["kind"] == "ferroelectric_effective_hamiltonian"

    parameters = _ferroelectric_parameter_arrays(record, jnp.float64)
    reference = record["reference_observable"]
    field = jnp.broadcast_to(
        jnp.asarray(reference["configuration"], dtype=jnp.float64),
        (3, 3, 3, 3),
    )
    energy = self_energy_onsite_isotropic(field, parameters["onsite"])
    energy += short_range_1stnn_isotropic(field, parameters["short_range_1"])
    energy += short_range_2ednn_isotropic(field, parameters["short_range_2"])
    energy += get_short_range_3rdnn_isotropic()(
        field, parameters["short_range_3"]
    )

    np.testing.assert_allclose(
        energy / 27,
        reference["value"],
        rtol=0,
        atol=reference["atol"],
    )
    assert parameters["onsite"].shape == (3,)
    assert parameters["ewald_prefactor"].shape == (1,)
    assert parameters["onsite"].dtype == jnp.float64


@pytest.mark.scientific
@pytest.mark.parametrize("path", MAGNETIC_RECORDS, ids=lambda path: path.stem)
def test_magnetic_record_matches_production_engines(path):
    record = _load_record(path)
    assert record["model"]["kind"] == "classical_heisenberg"

    lattice_class = {
        "body_centered_cubic": of.BodyCenteredCubic3D,
        "simple_cubic": of.SimpleCubic3D,
    }[record["lattice"]["type"]]
    lattice = lattice_class(4, 4, 4)
    parameters = record["parameters"]
    reference = record["reference_observable"]
    field = jnp.broadcast_to(
        jnp.asarray(reference["configuration"], dtype=jnp.float64)
        * parameters["moment_mu_B"],
        (4, 4, 4, 3),
    )
    couplings = _magnetic_exchange_couplings(record, jnp.float64)
    energy = jnp.asarray(0.0, dtype=jnp.float64)
    shell_names = ("first", "second", "third", "fourth")
    for shell_name, coupling in zip(shell_names, couplings):
        rollers = getattr(lattice, f"{shell_name}_shell_roller")
        engine = get_isotropic_exchange_energy_engine(rollers)
        energy += engine(field, jnp.asarray((coupling,)))

    np.testing.assert_allclose(
        energy / 64,
        reference["value"],
        rtol=0,
        atol=reference["atol"],
    )
    assert couplings.shape == (record["lattice"]["shells"],)
    assert couplings.dtype == jnp.float64


def test_ferroelectric_records_select_determinant_pressure_volume():
    for path in FERROELECTRIC_RECORDS:
        record = _load_record(path)
        assert record["conventions"]["pressure_volume"] == "determinant"


def test_exchange_records_declare_pair_counting_and_unit_conversion():
    bcc = _load_record(MAGNETIC_RECORDS[0])
    simple_cubic = _load_record(MAGNETIC_RECORDS[1])

    np.testing.assert_allclose(
        _magnetic_exchange_couplings(bcc, jnp.float64),
        2
        * np.asarray(bcc["parameters"]["exchange_source_values"])
        * Constants.mRy
        / bcc["parameters"]["moment_mu_B"] ** 2,
        rtol=1e-7,
    )
    np.testing.assert_allclose(
        _magnetic_exchange_couplings(simple_cubic, jnp.float64),
        np.asarray(simple_cubic["parameters"]["exchange_source_values"])
        * Constants.Joule
        / simple_cubic["parameters"]["moment_mu_B"] ** 2,
        rtol=1e-7,
    )


@pytest.mark.parametrize(
    "record_path, script_path",
    MAGNETIC_EXAMPLES,
    ids=("bcc_Fe", "simple_cubic"),
)
def test_magnetic_example_conversion_matches_record(record_path, script_path):
    record = _load_record(record_path)
    example = _load_example_module(script_path)

    np.testing.assert_allclose(
        example._exchange_couplings(record),
        _magnetic_exchange_couplings(record, jnp.float64),
        rtol=1e-7,
    )
