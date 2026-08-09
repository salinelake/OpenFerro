# OpenFerro examples

The maintained examples use explicit entry points and reproducible random
seeds. Examples 01 through 03 retain working-directory-relative configuration
and output defaults, so launch them from their own directories unless you pass
explicit paths; each provides a ``--tiny`` development check. Example 04
resolves its defaults from the repository and script locations and documents
an explicitly reduced smoke command. These short runs do not establish
production convergence.

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
