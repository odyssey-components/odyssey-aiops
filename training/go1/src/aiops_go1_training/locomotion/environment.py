"""Experimental Go1 MJX training environment; not the upstream SB3 environment."""

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
from mujoco import MjData, MjModel, mjx

EnvironmentState = tuple[mjx.Data, jax.Array, jax.Array]


def reward(quat: jax.Array, velocity: jax.Array, action: jax.Array, target_speed: float) -> jax.Array:
    upright = 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)
    tracking = jnp.exp(-((velocity[0] - target_speed) ** 2) / 0.25)
    return 0.02 * (tracking + upright - 0.02 * jnp.sum(action**2) - 0.1 * velocity[2] ** 2)


def has_fallen(height: float | jax.Array, quat: jax.Array) -> jax.Array:
    upright = 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)
    return (height < 0.19) | (height > 0.65) | (upright < 0.85)


def create_environment(model: MjModel, mjx_model: mjx.Model, initial: MjData, target_speed: float = 0.0):
    if (model.nq, model.nv, model.nu) != (19, 18, 12) or abs(model.opt.timestep - 0.002) > 1e-9:
        raise ValueError("unexpected Go1 model")
    initial_data = mjx.put_data(model, initial)
    default = jnp.asarray(initial.qpos[7:])
    lo, hi = jnp.asarray(model.actuator_ctrlrange[:, 0]), jnp.asarray(model.actuator_ctrlrange[:, 1])

    def reset() -> EnvironmentState:
        return initial_data, jnp.zeros(12), jnp.array(0, dtype=jnp.int32)

    def observe(data: mjx.Data, last: jax.Array) -> jax.Array:
        return jnp.concatenate((data.qpos[7:] - default, data.qvel, jnp.array([target_speed, 0.0, 0.0]), last))

    def step(state: EnvironmentState, action: jax.Array) -> tuple[EnvironmentState, jax.Array, jax.Array, jax.Array]:
        data, _, count = state
        action = jnp.clip(action, lo, hi)

        def physics(d: mjx.Data, _: None) -> tuple[mjx.Data, None]:
            return mjx.step(mjx_model, d.replace(ctrl=action)), None

        data, _ = jax.lax.scan(physics, data, None, length=10)
        fallen = has_fallen(data.qpos[2], data.qpos[3:7])
        score = reward(data.qpos[3:7], data.qvel[:3], action, target_speed) - jnp.where(fallen, 2.0, 0.0)
        count = count + 1
        done = fallen | (count >= 750)
        # Both branches produce the same typed MJX state pytree.
        select_state = cast(
            Callable[[jax.Array, Callable[[], EnvironmentState], Callable[[], EnvironmentState]], EnvironmentState],
            jax.lax.cond,
        )
        next_state = select_state(done, reset, lambda: (data, action, count))
        return next_state, score, done, data.qpos

    return reset, observe, step
