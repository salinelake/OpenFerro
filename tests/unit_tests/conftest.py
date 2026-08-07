import jax


# Scientific finite differences need a test-only precision baseline. This does
# not change OpenFerro's production default, which remains controlled by JAX.
jax.config.update("jax_enable_x64", True)
