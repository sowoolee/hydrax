from typing import Literal, Tuple

import jax
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)
import jax.numpy as jnp
from flax.struct import dataclass

from hydrax.alg_base import SamplingBasedController, SamplingParams, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task
from mujoco import mjx
from typing import Any, Literal, Tuple
import flax.nnx as nnx
@dataclass
class GAMDALFParams(SamplingParams):
    """Policy parameters for Diffusion-Inspired Annealing for Legged MPC (GAMDALF).

    Same as SamplingParams, but with a different name for clarity.

    Attributes:
        tk: The knot times of the control spline.
        mean: The mean of the control spline knot distribution, μ = [u₀, ...].
        rng: The pseudo-random number generator key.
        opt_iteration: The optimization iteration number.
    """

    opt_iteration: int


class GAMDALF(SamplingBasedController):
    """Diffusion-Inspired Annealing for Legged MPC (GAMDALF) based on https://arxiv.org/abs/2409.15610.

    GAMDALF-MPC is MPPI with a dual-loop, annealed sampling covariance that:
        - Decreases across optimisation iterations (trajectory-level annealing).
        - Increases along the planning horizon (action-level annealing).

    The noise level is given by:

        σ[i,h] = σ₀ * exp(-i/(β₁*N) - (H-h)/(β₂*H))

    where:
        - σ₀ is the tunable `noise_level`,
        - β₁ is the tunable `beta_opt_iter`,
        - β₂ is the tunable `beta_horizon`,
        - i in {0,...,N-1} is the optimisation iteration,
        - h in {0,...,H} indexes the knot along the horizon, and
        - N is the number of iterations and H is the number of knots.
    """

    def __init__(
        self,
        task: Task,
        num_samples: int,
        noise_level: float,
        beta_opt_iter: float,
        beta_horizon: float,
        temperature: float,
        num_randomizations: int = 1,
        risk_strategy: RiskStrategy = None,
        seed: int = 0,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 1,
    ) -> None:
        """Initialize the controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            num_samples: The number of control sequences to sample.
            beta_opt_iter: The temperature parameter β₁ for the trajectory-level
                          annealing. Higher values will result in less
                          annealing over optimisation iterations (exploration).
            beta_horizon: The temperature parameter β₂ for the action-level
                          annealing. Higher values will result in less
                          variation over the planning horizon (exploitation).
            temperature: The MPPI temperature parameter λ. Higher values take a more
                         even average over the samples.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combining costs from different randomizations.
                           Defaults to average cost.
            seed: The random seed for domain randomization.
            plan_horizon: The time horizon for the rollout in seconds.
            spline_type: The type of spline used for control interpolation.
                         Defaults to "zero" (zero-order hold).
            num_knots: The number of knots in the control spline.
        """
        super().__init__(
            task,
            num_randomizations=num_randomizations,
            risk_strategy=risk_strategy,
            seed=seed,
            plan_horizon=plan_horizon,
            spline_type=spline_type,
            num_knots=num_knots,
            iterations=iterations,
        )
        self.noise_level = noise_level
        self.beta_opt_iter = beta_opt_iter
        self.beta_horizon = beta_horizon
        assert self.beta_opt_iter > 0.0, "beta_opt_iter must be positive"
        assert self.beta_horizon > 0.0, "beta_horizon must be positive"
        self.num_samples = num_samples
        self.temperature = temperature

    def init_params(
        self, initial_knots: jax.Array = None, seed: int = 0
    ) -> GAMDALFParams:
        """Initialize the policy parameters."""
        _params = super().init_params(initial_knots, seed)

        return GAMDALFParams(
            tk=_params.tk, mean=_params.mean, rng=_params.rng, opt_iteration=0
            )

    def sample_knots(self, params: GAMDALFParams) -> Tuple[jax.Array, GAMDALFParams]:
        """Sample control knots.

        Anneals noise and adds it to the mean control sequence, then increments
        the optimisation iteration number.
        """
        rng, sample_rng = jax.random.split(params.rng)

        noise = jax.random.normal(
            sample_rng,
            (self.num_samples, self.num_knots, self.task.model.nu),
        )

        noise_level = self.noise_level * jnp.exp(
            -(params.opt_iteration) / (self.beta_opt_iter * self.iterations)
            - (self.num_knots - 1 - jnp.arange(self.num_knots))
            / (self.beta_horizon * self.num_knots)
        )

        controls = params.mean + noise_level[None, :, None] * noise

        # Increment opt_iteration, wrapping after maximum iterations reached
        return controls, params.replace(
            opt_iteration=(params.opt_iteration + 1) % self.iterations,
            rng=rng,
        )

    def optimize(self, state: mjx.Data, params: Any) -> Tuple[Any, Trajectory]:
        """Perform an optimization step to update the policy parameters.

        Args:
            state: The initial state x₀.
            params: The current policy parameters, U ~ π(params).

        Returns:
            Updated policy parameters
            Rollouts used to update the parameters
        """
        # Warm-start spline by advancing knot times by sim dt, then recomputing
        # the mean knots by evaluating the old spline at those times
        tk = params.tk
        new_tk = (
            jnp.linspace(0.0, self.plan_horizon, self.num_knots) + state.time
        )
        new_mean = self.interp_func(new_tk, tk, params.mean[None, ...])[0]
        params = params.replace(tk=new_tk, mean=new_mean)

        def _optimize_scan_body(params: Any, _: Any):
            # Sample random control sequences from spline knots
            knots, params = self.sample_knots(params)
            knots = jnp.clip(
                knots, self.task.u_min, self.task.u_max
            )  # (num_rollouts, num_knots, nu)
            # jax.debug.print("knots = {}", knots)
            # Roll out the control sequences, applying domain randomizations and
            # combining costs using self.risk_strategy.
            rng, dr_rng = jax.random.split(params.rng)
            # rollouts = self.rollout_with_randomizations(
            #     state, new_tk, knots, dr_rng
            # )
            def rollout_with_randomizations_for_grad(
                    knots: jax.Array,
                    state: mjx.Data,
                    tk: jax.Array,
                    rng: jax.Array,
            ):
                rollouts = self.rollout_with_randomizations(
                    state, tk, knots, rng
                )
                costs = jnp.sum(rollouts.costs, axis=1)
                # jax.debug.print("real costs = {}", -costs / self.temperature)
                weights = jnp.exp(-costs / self.temperature)
                # weights = jax.nn.softmax(-costs / self.temperature)
                # jax.debug.print("real weights = {}", weights)
                sum_weights = jnp.sum(weights[:, None, None], axis=0).reshape()
                # jax.debug.print("real sum_weights = {}", sum_weights)
                return sum_weights, rollouts
            (costs, rollouts), grads = nnx.value_and_grad(rollout_with_randomizations_for_grad, has_aux=True)(knots, state, new_tk, dr_rng)
            sum_grads = jnp.sum(grads, axis=0)
            jax.debug.print("params.mean = {}", params.mean)
            jax.debug.print("costs = {}", costs)
            jax.debug.print("sum_grads = {}", sum_grads)
            params = params.replace(rng=rng)
            params = params.replace(mean=params.mean+sum_grads/costs)

            # Update the policy parameters based on the combined costs
            # params = self.update_params(params, rollouts)

            return params, rollouts

        params, rollouts = jax.lax.scan(
            f=_optimize_scan_body, init=params, xs=jnp.arange(self.iterations)
        )

        rollouts_final = jax.tree.map(lambda x: x[-1], rollouts)

        return params, rollouts_final


    def update_params(
        self, params: GAMDALFParams, rollouts: Trajectory
    ) -> GAMDALFParams:
        """Update the mean with an exponentially weighted average."""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        # N.B. jax.nn.softmax takes care of details like baseline subtraction.
        weights = jax.nn.softmax(-costs / self.temperature, axis=0)
        mean = jnp.sum(weights[:, None, None] * rollouts.knots, axis=0)
        return params.replace(mean=mean)
