import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class CartPole(Task):
    """A cart-pole swingup task."""

    def __init__(self) -> None:
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/cart_pole/scene.xml"
        )
        super().__init__(mj_model, trace_sites=["tip"])

    def _distance_to_upright(self, state: mjx.Data) -> jax.Array:
        """Get a measure of distance to the upright position."""
        theta = state.qpos[1] + jnp.pi
        # state_theta = state.qpos[1]
        # raw = state_theta % (2 * jnp.pi)
        #
        # theta = jnp.where(
        #     (raw >= -0.5 * jnp.pi) & (raw <= 0.5 * jnp.pi),
        #     raw - jnp.pi,
        #     (state_theta - jnp.pi) % (2 * jnp.pi) + jnp.pi - jnp.pi
        # )
        theta_err = jnp.array([jnp.cos(theta) - 1, jnp.sin(theta)])
        #theta_err =  jnp.arctan2(jnp.sin(state_theta), jnp.cos(state_theta))- jnp.pi

        return jnp.sum(jnp.square(theta_err))

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        theta_cost = self._distance_to_upright(state)
        centering_cost = jnp.sum(jnp.square(state.qpos[0]))
        velocity_cost = 0.01 * jnp.sum(jnp.square(state.qvel))
        control_cost = 0.01 * jnp.sum(jnp.square(control))
        return theta_cost + centering_cost + velocity_cost + control_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        theta_cost = 10 * self._distance_to_upright(state)
        centering_cost = jnp.sum(jnp.square(state.qpos[0]))
        velocity_cost = 0.01 * jnp.sum(jnp.square(state.qvel))
        return theta_cost + centering_cost + velocity_cost
