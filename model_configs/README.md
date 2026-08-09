# Model records

The JSON files in this directory are documented parameter records, not inputs
to a closed OpenFerro model schema. OpenFerro deliberately does not expose a
model-specific configuration loader: lattice models are assembled through the
normal lattice, field, interaction, and simulation APIs, and callers decide how
their own data maps to those APIs.

Read a record with the standard library and extract only the sections needed by
the model being constructed:

```python
import json
from pathlib import Path

record = json.loads(
    Path("model_configs/BaTiO3_LDA.json").read_text(encoding="utf-8")
)
onsite = record["parameters"]["onsite"]
```

The maintained records use `schema_version: 1` to identify their current file
layout. This is record metadata, not a restriction on model kinds or parameter
sections accepted by OpenFerro. Other lattice models may use additional fields,
different parameter groups, or their own serialization format.

## Maintained metadata

The current records carry:

- `model`: stable ID, material, method, free-form model kind, citation, and DOI;
- `lattice`: geometry needed to reproduce the recorded model;
- `units`: units of source values and the intended engine inputs;
- `conventions`: strain, geometry, or pair-counting decisions;
- `parameters`: model-specific parameter groups;
- `reference_observable`: a small independently reproducible value and
  tolerance.

`tests/unit_tests/test_model_records.py` checks that every maintained record has
provenance and finite numerical data. It also maps the known records to the
relevant production engines and recomputes their reference observables. These
are tests of the shipped records, not validation rules imposed on arbitrary
user models.

## Ferroelectric records

The maintained ferroelectric records describe an axis-aligned orthogonal cell,
soft-mode displacements in Angstrom, energies in eV, engineering Voigt strain
ordered as `(exx, eyy, ezz, 2eyz, 2exz, 2exy)`, and unique short-range bonds.
They select determinant pressure volume, `V0 * det(I + strain)`. The BTO example
performs the explicit mapping from the named onsite, short-range, elastic,
elastic-dipole, and Born-charge values to OpenFerro calls.

For a linearized-volume comparison in a custom setup, pass
`pressure_volume="linearized_small_strain"` to `System.add_global_strain`.
The BTO example intentionally uses the determinant default and does not expose
a pressure-volume command-line option. Changing record metadata alone does not
reconfigure a system because these records are not loaded through a package
schema.

## Magnetic records

The magnetic example records keep exchange values in their source units and
declare source and engine pair-counting conventions. The example scripts apply
the conversion immediately before registering interactions. An ordered-pair
source such as bcc Fe uses

```text
J_engine = 2 * J_source * unit_to_eV / moment_mu_B**2
```

while a unique-bond source omits the factor of two. The public exchange engine
also retains `bond_counting="ordered"` for parameters calibrated to the legacy
OpenFerro convention. Do not both double a coupling and select `ordered`.

## Extending the collection

A new record should make its provenance, units, conventions, and at least one
reference observable inspectable, but it does not need to fit either existing
parameter layout. Add model-specific extraction beside the example or
application that uses it. A future general serialization API should be designed
around extensibility or registration rather than adding new hard-coded model
branches to the OpenFerro package.
