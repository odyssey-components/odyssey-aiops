"""Depth CNN plus proprioception: Gaussian actor/critic and clipped PPO."""

from typing import NamedTuple, cast

import jax
import jax.numpy as jnp

from ..upstream.policy import forward as base_forward

Params = dict[str, jax.Array]


class Adam(NamedTuple):
    step: jax.Array
    m: Params
    v: Params


def initialize(key: jax.Array, height: int, width: int, *, base: Params | None = None) -> Params:
    keys = jax.random.split(key, 6)
    params: Params = {"log_std": jnp.full((12,), -0.7)}
    for index, (inputs, outputs) in enumerate(((1, 8), (8, 16), (16, 16))):
        params[f"c{index + 1}"] = jax.random.normal(keys[index], (3, 3, inputs, outputs)) / jnp.sqrt(float(9 * inputs))
        params[f"b{index + 1}"] = jnp.zeros(outputs)
    features = ((height + 7) // 8) * ((width + 7) // 8) * 16 + 45
    for index, (name, inputs, outputs, scale) in enumerate(
        (("dense", features, 64, 1.0), ("actor", 64, 12, 0.01), ("critic", 64, 1, 1.0))
    ):
        params[name] = jax.random.normal(keys[index + 3], (inputs, outputs)) * (scale / jnp.sqrt(float(inputs)))
        params[name + "_bias"] = jnp.zeros(outputs)
    if base is not None:
        params["actor"] = jnp.zeros_like(params["actor"])
        params["log_std"] = base["log_std"]
        params.update({"base." + name: value for name, value in base.items()})
    return params


def forward(params: Params, depth: jax.Array, state: jax.Array) -> tuple[jax.Array, jax.Array]:
    features = depth
    for index in range(1, 4):
        features = jax.nn.relu(
            jax.lax.conv_general_dilated(
                features, params[f"c{index}"], (2, 2), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
            )
            + params[f"b{index}"]
        )
    fused = jnp.concatenate((features.reshape((features.shape[0], -1)), state), axis=-1)
    hidden = jnp.tanh(fused @ params["dense"] + params["dense_bias"])
    mean = hidden @ params["actor"] + params["actor_bias"]
    if "base.log_std" in params:
        base = {
            name.removeprefix("base."): jax.lax.stop_gradient(value)
            for name, value in params.items()
            if name.startswith("base.")
        }
        base_mean, _ = base_forward(base, state)
        mean = mean + base_mean
    return mean, (hidden @ params["critic"] + params["critic_bias"]).squeeze(-1)


def distribution_log_std(params: Params) -> jax.Array:
    return params["log_std"] if "base.log_std" in params else jnp.clip(params["log_std"], -5.0, 2.0)


def log_probability(params: Params, mean: jax.Array, actions: jax.Array) -> jax.Array:
    log_std = distribution_log_std(params)
    return (-0.5 * (((actions - mean) / jnp.exp(log_std)) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi))).sum(axis=-1)


def adam(params: Params) -> Adam:
    zeros = cast(Params, jax.tree.map(jnp.zeros_like, params))
    return Adam(jnp.array(0), zeros, zeros)


def update(
    params: Params,
    optimizer: Adam,
    depth: jax.Array,
    state: jax.Array,
    actions: jax.Array,
    old_logp: jax.Array,
    advantages: jax.Array,
    returns: jax.Array,
    rate: float,
) -> tuple[Params, Adam, dict[str, jax.Array]]:
    def loss(candidate: Params) -> jax.Array:
        mean, values = forward(candidate, depth, state)
        ratio = jnp.exp(log_probability(candidate, mean, actions) - old_logp)
        normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        actor = -jnp.minimum(ratio * normalized, jnp.clip(ratio, 0.8, 1.2) * normalized).mean()
        critic = 0.5 * jnp.mean((values - returns) ** 2)
        entropy = jnp.sum(distribution_log_std(candidate) + 0.5 * (1 + jnp.log(2 * jnp.pi)))
        return actor + 0.5 * critic - 0.001 * entropy

    value, gradients = cast(tuple[jax.Array, Params], jax.value_and_grad(loss)(params))  # pyright: ignore[reportUnknownMemberType]
    norm = jnp.sqrt(sum(jnp.sum(gradient**2) for gradient in gradients.values()))
    gradients = {name: gradient * jnp.minimum(1.0, 0.5 / (norm + 1e-8)) for name, gradient in gradients.items()}
    count = optimizer.step + 1
    first = {name: 0.9 * optimizer.m[name] + 0.1 * gradient for name, gradient in gradients.items()}
    second = {name: 0.999 * optimizer.v[name] + 0.001 * gradient**2 for name, gradient in gradients.items()}
    updated = {
        name: parameter - rate * (first[name] / (1 - 0.9**count)) / (jnp.sqrt(second[name] / (1 - 0.999**count)) + 1e-8)
        for name, parameter in params.items()
    }
    return updated, Adam(count, first, second), {"loss": value, "gradient_norm": norm}
