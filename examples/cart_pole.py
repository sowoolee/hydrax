import argparse

import mujoco

from hydrax.algs import CEM, MPPI, PredictiveSampling, DIAL, GAMDALF
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.cart_pole import CartPole

"""
Run an interactive simulation of a cart-pole swingup
"""

# Define the task (cost and dynamics)
task = CartPole()

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run an interactive simulation of the cube rotation task."
)
subparsers = parser.add_subparsers(
    dest="algorithm", help="Sampling algorithm (choose one)"
)
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser("cem", help="Cross-Entropy Method")
subparsers.add_parser("dial", help="Diffusion-Inspired Annealing for Legged MPC (DIAL)")
subparsers.add_parser("gamdalf", help="Gradient-Aided Model-based Diffusion accelerated by Newton–Langevin Flow (GAMDaLF)")
args = parser.parse_args()

# Set up the controller
if args.algorithm == "ps" or args.algorithm is None:
    print("Running predictive sampling")
    ctrl = PredictiveSampling(
        task,
        num_samples=128,
        noise_level=0.3,
        spline_type="cubic",
        plan_horizon=1.0,
        num_knots=4,
    )
elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task,
        num_samples=128,
        noise_level=0.3,
        temperature=0.1,
        spline_type="cubic",
        plan_horizon=1.0,
        num_knots=4,
    )
elif args.algorithm == "cem":
    print("Running CEM")
    ctrl = CEM(
        task,
        num_samples=128,
        num_elites=3,
        sigma_start=0.5,
        sigma_min=0.1,
        spline_type="cubic",
        plan_horizon=1.0,
        num_knots=4,
    )
elif args.algorithm == "dial":
    print("Running Diffusion-Inspired Annealing for Legged MPC (DIAL)")
    ctrl = DIAL(
        task,
        num_samples=4,
        noise_level=0.4,
        beta_opt_iter=1.0,
        beta_horizon=1.0,
        temperature=0.01,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=11,
        iterations=5,
    )
elif args.algorithm == "gamdalf":
    print("Gradient-Aided Model-based Diffusion accelerated by Newton–Langevin Flow (GAMDaLF)")
    ctrl = GAMDALF(
        task,
        num_samples=4,
        noise_level=0.4,
        beta_opt_iter=1,
        beta_horizon=1,
        temperature=0.001,
        plan_horizon=1,
        spline_type="cubic",
        num_knots=10,
        iterations=5,
        seed=1,
    )
else:
    parser.error("Other algorithms not implemented for this example!")

# Define the model used for simulation
mj_model = task.mj_model
mj_data = mujoco.MjData(mj_model)

# Run the interactive simulation
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=50,
    # fixed_camera_id=0,
    show_traces=False,
    max_traces=1,
)
