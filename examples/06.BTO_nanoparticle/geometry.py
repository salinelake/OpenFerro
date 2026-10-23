"""Host-side geometry helpers for the BaTiO3 nanoparticle example."""

from dataclasses import dataclass
import math

import numpy as np


FACE_NAMES = ("x-", "x+", "y-", "y+", "z-", "z+")
CORNER_OFFSETS = np.asarray(
    [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ],
    dtype=np.int32,
)


@dataclass(frozen=True)
class ParticleGeometry:
    """Validated spherical active-cell geometry in a padded cubic box."""

    box_size: int
    radius_cells: float
    center: tuple
    cell_mask: np.ndarray
    active_indices: np.ndarray
    relative_coordinates: np.ndarray
    n_particle: int
    face_padding: tuple
    reference_volume: float
    acoustic_mask: np.ndarray
    acoustic_indices: np.ndarray
    acoustic_relative_coordinates: np.ndarray
    cell_node_indices: np.ndarray
    constraint_basis: np.ndarray
    n_acoustic: int

    @property
    def face_padding_by_name(self):
        """Return inactive face-layer counts keyed by face name."""
        return dict(zip(FACE_NAMES, self.face_padding))


def _validate_center(center, box_size):
    if center is None:
        if box_size % 2 == 0:
            raise ValueError(
                "Even box sizes require an explicit integer or half-cell center."
            )
        return ((box_size - 1) // 2,) * 3

    center_array = np.asarray(center, dtype=float)
    if center_array.shape != (3,) or not np.all(np.isfinite(center_array)):
        raise ValueError("center must contain three finite coordinates.")
    doubled_center = 2.0 * center_array
    if not np.allclose(
        doubled_center, np.rint(doubled_center), rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("center coordinates must be integers or half-integers.")
    center_array = np.rint(doubled_center) / 2.0
    if np.any(center_array < 0.0) or np.any(center_array > box_size - 1):
        raise ValueError("center must lie within the padded lattice index range.")
    return tuple(float(value) for value in center_array)


def build_spherical_geometry(
    box_size,
    radius_cells,
    lattice_constant,
    *,
    center=None,
    required_inactive_layers=1,
):
    """Build and validate a site mask for a spherical particle.

    Parameters
    ----------
    box_size : int
        Number of effective-Hamiltonian cells along every padded-box axis.
    radius_cells : float
        Particle radius in units of the cubic reference lattice constant.
    lattice_constant : float
        Cubic reference lattice constant in Angstrom.
    center : sequence of float, optional
        Center in lattice-index coordinates.  The default is the central site
        of an odd box.  Even boxes require this argument explicitly.
    required_inactive_layers : int, optional
        Minimum number of fully inactive index layers between the active mask
        and each periodic face.

    Returns
    -------
    ParticleGeometry
        Host-resident immutable geometry metadata.  Array contents are marked
        read-only before return.
    """
    if (
        isinstance(box_size, (bool, np.bool_))
        or not isinstance(box_size, (int, np.integer))
        or box_size <= 0
    ):
        raise ValueError("box_size must be a positive integer.")
    box_size = int(box_size)
    if (
        isinstance(required_inactive_layers, (bool, np.bool_))
        or not isinstance(required_inactive_layers, (int, np.integer))
        or required_inactive_layers < 0
    ):
        raise ValueError("required_inactive_layers must be a non-negative integer.")
    if not np.isscalar(radius_cells) or not math.isfinite(float(radius_cells)):
        raise ValueError("radius_cells must be a finite non-negative scalar.")
    radius_cells = float(radius_cells)
    if radius_cells < 0.0:
        raise ValueError("radius_cells must be a finite non-negative scalar.")
    if not np.isscalar(lattice_constant) or not math.isfinite(float(lattice_constant)):
        raise ValueError("lattice_constant must be finite and strictly positive.")
    lattice_constant = float(lattice_constant)
    if lattice_constant <= 0.0:
        raise ValueError("lattice_constant must be finite and strictly positive.")

    center = _validate_center(center, box_size)
    grid = np.indices((box_size, box_size, box_size), dtype=float)
    squared_distance = sum(
        (grid[axis] - center[axis]) ** 2 for axis in range(3)
    )
    cell_mask = squared_distance <= radius_cells**2 + 1.0e-12
    active_indices = np.argwhere(cell_mask)
    n_particle = int(active_indices.shape[0])
    if n_particle == 0:
        raise ValueError("The spherical geometry contains no active cells.")

    minima = active_indices.min(axis=0)
    maxima = active_indices.max(axis=0)
    face_padding = (
        int(minima[0]),
        int(box_size - 1 - maxima[0]),
        int(minima[1]),
        int(box_size - 1 - maxima[1]),
        int(minima[2]),
        int(box_size - 1 - maxima[2]),
    )
    if min(face_padding) < required_inactive_layers:
        details = ", ".join(
            f"{name}={layers}" for name, layers in zip(FACE_NAMES, face_padding)
        )
        raise ValueError(
            f"Particle needs at least {required_inactive_layers} inactive layer(s) "
            f"on every periodic face; found {details}."
        )

    relative_coordinates = active_indices.astype(float) - np.asarray(center)
    reference_volume = n_particle * lattice_constant**3

    cell_nodes = (
        active_indices[:, None, :].astype(np.int32)
        + CORNER_OFFSETS[None, :, :]
    )
    if np.any(cell_nodes < 0) or np.any(cell_nodes >= box_size):
        raise ValueError(
            "Occupied cells need a nonperiodic high-face halo for acoustic "
            "corner nodes."
        )
    flat_cell_nodes = (
        (cell_nodes[..., 0] * box_size + cell_nodes[..., 1]) * box_size
        + cell_nodes[..., 2]
    ).astype(np.int32)
    acoustic_mask = np.zeros(cell_mask.shape, dtype=bool)
    acoustic_mask.reshape(-1)[flat_cell_nodes.reshape(-1)] = True
    acoustic_indices = np.argwhere(acoustic_mask).astype(np.int32)
    n_acoustic = int(acoustic_indices.shape[0])
    acoustic_relative_coordinates = (
        acoustic_indices.astype(float)
        - 0.5
        - np.asarray(center, dtype=float)
    )
    active_basis = np.column_stack(
        (
            np.ones(n_acoustic, dtype=float),
            acoustic_relative_coordinates,
        )
    )
    if np.linalg.matrix_rank(active_basis) != 4:
        raise ValueError(
            "Acoustic constraint basis is rank deficient; the particle must "
            "span three dimensions."
        )
    constraint_basis = np.zeros(cell_mask.shape + (4,), dtype=float)
    constraint_basis[acoustic_mask] = active_basis

    cell_mask.setflags(write=False)
    active_indices.setflags(write=False)
    relative_coordinates.setflags(write=False)
    acoustic_mask.setflags(write=False)
    acoustic_indices.setflags(write=False)
    acoustic_relative_coordinates.setflags(write=False)
    flat_cell_nodes.setflags(write=False)
    constraint_basis.setflags(write=False)
    return ParticleGeometry(
        box_size=box_size,
        radius_cells=radius_cells,
        center=center,
        cell_mask=cell_mask,
        active_indices=active_indices,
        relative_coordinates=relative_coordinates,
        n_particle=n_particle,
        face_padding=face_padding,
        reference_volume=reference_volume,
        acoustic_mask=acoustic_mask,
        acoustic_indices=acoustic_indices,
        acoustic_relative_coordinates=acoustic_relative_coordinates,
        cell_node_indices=flat_cell_nodes,
        constraint_basis=constraint_basis,
        n_acoustic=n_acoustic,
    )


def radial_surface_shell_mask(geometry, shell_thickness_cells):
    """Return the active cells in a fixed-thickness radial surface shell.

    Parameters
    ----------
    geometry : ParticleGeometry
        Spherical particle geometry.
    shell_thickness_cells : float
        Positive shell thickness in reference-lattice cell units. It must be
        smaller than the particle radius so that a core remains.

    Returns
    -------
    numpy.ndarray
        Read-only Boolean mask with the same shape as ``geometry.cell_mask``.
    """
    if (
        not np.isscalar(shell_thickness_cells)
        or isinstance(shell_thickness_cells, (bool, np.bool_))
        or not np.isfinite(float(shell_thickness_cells))
    ):
        raise ValueError("shell_thickness_cells must be a finite scalar.")
    shell_thickness_cells = float(shell_thickness_cells)
    if shell_thickness_cells <= 0.0:
        raise ValueError("shell_thickness_cells must be strictly positive.")
    if shell_thickness_cells >= geometry.radius_cells:
        raise ValueError(
            "shell_thickness_cells must be smaller than the particle radius."
        )

    radii = np.linalg.norm(geometry.relative_coordinates, axis=1)
    shell_active = radii > geometry.radius_cells - shell_thickness_cells
    shell = np.zeros_like(geometry.cell_mask, dtype=bool)
    shell[tuple(np.asarray(geometry.active_indices).T)] = shell_active
    if not np.any(shell):
        raise ValueError("The requested thickness selects no surface cells.")
    shell.setflags(write=False)
    return shell