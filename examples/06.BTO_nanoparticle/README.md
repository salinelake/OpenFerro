# BaTiO3 nanoparticle analysis

Run the analysis from this directory on an allocated compute node:

```bash
python 02.analyze.py --output-dir output
```

The analysis reads `config.json`, `sample_dipole_avg.log`, and
`sample_thermo.log` from the simulation output directory, using the material
parameters in the example's `BaTiO3.json`. It saves the mean polarization and
ionic soft-mode susceptibility in `polarization_chi.json`, and the polarization
trajectory and autocorrelation in `polarization.png`. A run configured with
zero sampling steps writes a summary with empty polarization/chi results.

When `traj/sample_dipole_dump_*.npy` files are available, it also writes
`radial_soft_mode_profile.json` and `dipole_snapshots.png`. The snapshot plot
compares the first and last sampling frames. Without dumps, these two outputs
are skipped with a message. Enable dumps in `01.npt.py` with a positive
`--sample-dump-interval`.

A bulk comparison is optional:

```bash
python 02.analyze.py --output-dir output \
    --bulk-reference ../01.BTO_Cooling/output/field_270K_avg.log
```

The reference CSV contains `Step, ux, uy, uz`, with spatial-average soft modes
in Angstrom. Its time-averaged vector magnitude supplies snapshot colorbar
markers and expands the color range when needed. The reference file is read
only when snapshot dumps are available; no bulk trajectory is loaded by
default. Polarization and susceptibility use the nanoparticle sampling data.

The numerical summary is saved before plotting. Missing explicitly requested
references and malformed snapshots raise errors without discarding that
summary. Sampling logs are required whenever sampling steps were configured.
