import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding

from openferro.parallelism import DeviceMesh


def test_device_mesh_accepts_explicit_devices():
    devices = jax.devices()[:1]
    mesh = DeviceMesh(devices=devices, num_rows=1, num_cols=1)

    assert list(mesh.mesh.devices.flat) == devices
    assert isinstance(mesh.partition_sharding(), NamedSharding)
    assert isinstance(mesh.replicate_sharding(), NamedSharding)


def test_device_mesh_infers_single_device_shape():
    mesh = DeviceMesh(devices=jax.devices()[:1])

    assert mesh.mesh.shape == {"x": 1, "y": 1}


def test_device_mesh_preserves_shaped_devices():
    devices = np.array(jax.devices()[:1], dtype=object).reshape((1, 1))
    mesh = DeviceMesh(devices=devices)

    assert mesh.mesh.shape == {"x": 1, "y": 1}


def test_device_mesh_rejects_mismatched_shape():
    try:
        DeviceMesh(devices=jax.devices()[:1], num_rows=1, num_cols=2)
    except ValueError as exc:
        assert "number of devices" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched mesh shape.")


def test_partition_sharding_device_put():
    mesh = DeviceMesh(devices=jax.devices()[:1], num_rows=1, num_cols=1)
    values = jnp.zeros((2, 2, 2, 3))

    sharded = jax.device_put(values, mesh.partition_sharding())

    assert sharded.sharding == mesh.partition_sharding()


if __name__ == "__main__":
    test_device_mesh_accepts_explicit_devices()
    test_device_mesh_infers_single_device_shape()
    test_device_mesh_preserves_shaped_devices()
    test_device_mesh_rejects_mismatched_shape()
    test_partition_sharding_device_put()
