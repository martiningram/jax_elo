import jax.numpy as jnp
from jax import jit
from functools import partial
import numpy as np


def pos_def_mat_from_tri_elts(elts, mat_size, jitter=1e-6):

    cov_mat = lo_tri_from_elements(elts, mat_size)
    cov_mat = cov_mat @ cov_mat.T

    cov_mat = cov_mat + jnp.eye(mat_size) * jitter

    return cov_mat


def num_mat_elts(num_triangular_elts):

    sqrt_term = np.sqrt(8 * num_triangular_elts + 1)

    return ((sqrt_term - 1) / 2).astype(int)


def num_triangular_elts(mat_size, include_diagonal=True):

    if include_diagonal:
        return int(mat_size * (mat_size + 1) / 2)
    else:
        return int(mat_size * (mat_size - 1) / 2)


def lo_tri_from_elements(elements, n):

    L = jnp.zeros((n, n))
    indices = jnp.tril_indices(L.shape[0])
    L = L.at[indices].set(elements)

    return L
