"""PPO updates for separate actor/critic; reuse the existing numeric Adam convention."""

from typing import cast

import jax
import jax.numpy as jnp

from ..vision.policy import Adam
from .policy import Params, forward, log_probability


def update(
    params: Params,
    optimizer: Adam,
    body: jax.Array,
    actions: jax.Array,
    old_logp: jax.Array,
    advantages: jax.Array,
    returns: jax.Array,
    rate: float,
    teacher: Params | None = None,
) -> tuple[Params, Adam, dict[str, jax.Array]]:
    def loss(candidate: Params) -> jax.Array:
        mean, value = forward(candidate, body)
        ratio = jnp.exp(log_probability(candidate, mean, actions) - old_logp)
        normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        actor = -jnp.minimum(ratio * normalized, jnp.clip(ratio, 0.8, 1.2) * normalized).mean()
        entropy = jnp.sum(candidate["log_std"] + 0.5 * (1 + jnp.log(2 * jnp.pi)))
        penalty = 0.0
        if teacher is not None:
            reference, _ = forward(teacher, body)
            straight = (body[:, 30] > 0) & (body[:, 32] == 0)
            deviation = jnp.mean((mean - reference) ** 2, axis=1)
            penalty = 2.0 * jnp.sum(straight * deviation) / jnp.maximum(1, jnp.sum(straight))
        return actor + 0.25 * jnp.mean((value - returns) ** 2) - 0.001 * entropy + penalty

    value, gradients = cast(tuple[jax.Array, Params], jax.value_and_grad(loss)(params))  # pyright: ignore[reportUnknownMemberType]
    norm = jnp.sqrt(sum(jnp.sum(x * x) for x in gradients.values()))
    gradients = {k: x * jnp.minimum(1.0, 0.5 / (norm + 1e-8)) for k, x in gradients.items()}
    count = optimizer.step + 1
    first = {k: 0.9 * optimizer.m[k] + 0.1 * g for k, g in gradients.items()}
    second = {k: 0.999 * optimizer.v[k] + 0.001 * g * g for k, g in gradients.items()}
    updated = {
        k: p - rate * (first[k] / (1 - 0.9**count)) / (jnp.sqrt(second[k] / (1 - 0.999**count)) + 1e-8)
        for k, p in params.items()
    }
    return updated, Adam(count, first, second), {"loss": value}
