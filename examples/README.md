# OpenFerro examples

The maintained examples use explicit entry points, reproducible random seeds,
and paths resolved relative to their scripts. Examples 01 through 04 provide a
``--tiny`` execution mode for development checks; those short runs do not
establish production convergence.

| Example | Description | Status |
| --- | --- | --- |
| [01.BTO_Cooling](01.BTO_Cooling) | BaTiO3 cooling with Ewald, strain, minimization, and NPT Langevin dynamics. | Stable entry point |
| [02.bcc_Fe_Heating](02.bcc_Fe_Heating) | Four-shell bcc Fe Heisenberg heating with stochastic SIB dynamics. | Stable entry point |
| [03.sc_Ising_Heating](03.sc_Ising_Heating) | Simple-cubic continuous Heisenberg heating; the directory name is historical. | Stable entry point |
| [04.PTOSTO_superlattice](04.PTOSTO_superlattice) | Single-GPU field-driven PTO/STO domain dynamics and visualization. | Runnable; superlattice engines experimental |
| [05.BTO_GPU_Parallel](05.BTO_GPU_Parallel) | Single-node BTO scaling and memory benchmark on 1, 2, 3, and 4 GPUs. | Experimental multi-device benchmark |

Each directory documents its physical model, run commands, and outputs. Review
the repository [feature-status matrix](../docs/source/feature_status.rst) before
treating an example as scientific validation of every engine it exercises.
