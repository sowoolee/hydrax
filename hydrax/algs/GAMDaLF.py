from typing import Literal, Tuple

import jax
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)
# jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
from flax.struct import dataclass

from hydrax.alg_base import SamplingBasedController, SamplingParams, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task
from mujoco import mjx
from typing import Any, Literal, Tuple
# import flax.nnx as nnx
from jax import value_and_grad
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

    def _objective_for_grad(
            self,
            knots: jax.Array,
            state: mjx.Data,
            tk: jax.Array,
            rng: jax.Array,
    ):
        """knots에 대해 grad를 구할 scalar loss 함수.

        - rollout → costs 계산
        - softmax(-cost/λ)로 weight 계산
        - weight는 stop_gradient
        - loss = - 1/λ * Σ w_i * J_i
        """
        rollouts = self.rollout_with_randomizations(state, tk, knots, rng)
        # (num_samples, H+1) → time sum
        costs = jnp.sum(rollouts.costs, axis=1)  # (num_samples,)
        weights = jax.nn.softmax(-costs / self.temperature, axis=0)  # (num_samples,)

        # weight에는 grad 안 흐르게
        weights = jax.lax.stop_gradient(weights)

        # scalar loss = Σ w_i * J_i
        loss = jnp.sum(-weights * costs / self.temperature)

        # 필요하면 weights도 aux로 넘겨서 debug
        return loss, (rollouts, costs, weights)


    def optimize(self, state: mjx.Data, params: GAMDALFParams) -> Tuple[GAMDALFParams, Trajectory]:
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

            # Roll out the control sequences, applying domain randomizations and
            # combining costs using self.risk_strategy.
            rng, dr_rng = jax.random.split(params.rng) ## 여기까진 동일

            # gradient 계산: loss(knots) wrt knots
            (loss, (rollouts, costs, weights)), grads = jax.value_and_grad(
                lambda k: self._objective_for_grad(k, state, new_tk, dr_rng),
                has_aux=True,
            )(knots)

            max_norm = 1e6
            # grads = jnp.nan_to_num(grads, nan=0.0, posinf=max_norm, neginf=-max_norm)
            # jax.debug.print("GAMDALF grads before clipping: {}", grads.flatten())
            sum_grads = jnp.sum(grads, axis=0)
            # sum_grads = jnp.mean(grads, axis=0)
            # jax.debug.print("GAMDALF grads sum (score of p1): {}", sum_grads.flatten())
            # jax.debug.print("GAMDALF weights: {}", weights.flatten())
            # jax.debug.print("GAMDALF costs: {}", costs.flatten())


            noise_level = self.noise_level * jnp.exp(
                -(params.opt_iteration) / (self.beta_opt_iter * self.iterations)
                - (self.num_knots - 1 - jnp.arange(self.num_knots))
                / (self.beta_horizon * self.num_knots)
            )
            # step = noise_level[:, None] * sum_grads  # (num_knots, nu)
            lr = 0.001
            step = lr * sum_grads  # (num_knots, nu)

            jax.debug.print("iter = {} ,step = {}", params.opt_iteration ,step.flatten())
            # jax.debug.print("noise_levle = {}", noise_level.flatten())

            new_mean = jnp.clip(params.mean + step, self.task.u_min, self.task.u_max)
            params = params.replace(mean= new_mean)
            # jax.debug.print("GAMDALF new mean = {}", new_mean.flatten())
            # lam = 1e-3
            # d = u_del.shape[0]
            #
            # # Gram matrix: g g^T
            # A = jnp.outer(u_del, u_del) + lam * jnp.eye(d)
            #
            # # (A)^{-1} g  를 구하고 싶다면:
            # u_del = jnp.linalg.solve(A, u_del)  # shape: (d,)
            #
            # params = params.replace(mean=params.mean+u_del)
            # new_mean = jnp.clip(params.mean, self.task.u_min, self.task.u_max)
            # params = params.replace(mean=new_mean)

            params = params.replace(rng=rng)

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
