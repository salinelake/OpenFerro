import jax.numpy as jnp
from jax import jit 

def get_func(n):
    a = jnp.arange(n)
    def func(x):
        return (a * x).sum()
    return func

f = jit(get_func(3))

