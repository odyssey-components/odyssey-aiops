"""SB3 default separate 64/64 tanh actor/critic and its Go1 observation protocol."""

import jax
import jax.numpy as jnp

Params = dict[str, jax.Array]
SHAPES = {
    f"{branch}{i}.{part}": shape
    for branch in ("pi", "vf")
    for i, sizes in enumerate(((45, 64), (64, 64), (64, 12 if branch == "pi" else 1)))
    for part, shape in (("weight", sizes), ("bias", (sizes[1],)))
} | {"log_std": (12,)}


def observation(body: jax.Array) -> jax.Array:
    """Raw 45-vector with already selected previous-action timing; never mutate physics."""
    scale = jnp.ones(45).at[12:15].set(jnp.array([2.0, 2.0, 0.25])).at[30:33].set(jnp.array([2.0, 2.0, 0.25]))
    return jnp.clip(body * scale, -100.0, 100.0)


def forward(params: Params, body: jax.Array) -> tuple[jax.Array, jax.Array]:
    obs = observation(body)
    outputs: list[jax.Array] = []
    for branch in ("pi", "vf"):
        hidden = obs
        for i in range(3):
            hidden = hidden @ params[f"{branch}{i}.weight"] + params[f"{branch}{i}.bias"]
            if i != 2:
                hidden = jnp.tanh(hidden)
        outputs.append(hidden)
    return outputs[0], outputs[1].squeeze(-1)


def log_probability(params: Params, mean: jax.Array, actions: jax.Array) -> jax.Array:
    log_std = params["log_std"]
    return (-0.5 * (((actions - mean) / jnp.exp(log_std)) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi))).sum(axis=-1)
