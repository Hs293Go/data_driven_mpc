""" Implementation of the nonlinear optimizer for the data-augmented MPC.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""


import os

import casadi as cs
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from src.quad_mpc.quad_3d import Quadrotor3D
from src.utils.quad_3d_opt_utils import discretize_dynamics_and_cost
from src.utils.utils import (
    quaternion_inverse,
    safe_mkdir_recursive,
    skew_symmetric,
    quaternion_rotate_point,
    quaternion_product,
)
from .utils import (
    make_acados_optimizer,
    set_reference_trajectory,
    set_reference_state,
    optimize,
)


class Quad3DOptimizer:
    def __init__(
        self,
        quad: Quadrotor3D,
        t_horizon=1,
        n_nodes=20,
        q_cost=None,
        r_cost=None,
        q_mask=None,
        model_name="quad_3d_acados_mpc",
        solver_options=None,
    ):
        """
        :param quad: quadrotor object
        :type quad: Quadrotor3D
        :param t_horizon: time horizon for MPC optimization
        :param n_nodes: number of optimization nodes until time horizon
        :param q_cost: diagonal of Q matrix for LQR cost of MPC cost function. Must be a numpy array of length 12.
        :param r_cost: diagonal of R matrix for LQR cost of MPC cost function. Must be a numpy array of length 4.
        :param q_mask: Optional boolean mask that determines which variables from the state compute towards the cost
        function. In case no argument is passed, all variables are weighted.
        :param solver_options: Optional set of extra options dictionary for solvers.
        if not used
        """

        self.T = t_horizon  # Time horizon
        self.N = n_nodes  # number of control nodes within horizon

        self.quad = quad

        # Ensure current working directory is current folder
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        acados_models_dir = "../../acados_models"
        safe_mkdir_recursive(os.path.join(os.getcwd(), acados_models_dir))

        # Compile acados OCP solver if necessary
        json_file = os.path.join(acados_models_dir, model_name + "_acados_ocp.json")
        ocp, solver_data = make_acados_optimizer(
            t_horizon,
            n_nodes,
            q_cost,
            r_cost,
            q_mask,
            model_name,
            solver_options,
        )
        self.acados_ocp_solver = AcadosOcpSolver(
            ocp,
            json_file=json_file,
        )

    def set_reference_state(self, x_target=None, u_target=None):
        """
        Sets the target state and pre-computes the integration dynamics with cost equations
        :param x_target: 13-dimensional target state (p_xyz, a_wxyz, v_xyz, r_xyz)
        :param u_target: 4-dimensional target control input vector (u_1, u_2, u_3, u_4)
        """

        return set_reference_state(self.acados_ocp_solver, self.N, x_target, u_target)

    def set_reference_trajectory(self, x_target, u_target):
        """
        Sets the reference trajectory and pre-computes the cost equations for each point in the reference sequence.
        :param x_target: Nx13-dimensional reference trajectory (p_xyz, angle_wxyz, v_xyz, rate_xyz). It is passed in the
        form of a 4-length list, where the first element is a Nx3 numpy array referring to the position targets, the
        second is a Nx4 array referring to the quaternion, two more Nx3 arrays for the velocity and body rate targets.
        :param u_target: Nx4-dimensional target control input vector (u1, u2, u3, u4)
        """
        return set_reference_trajectory(
            self.acados_ocp_solver, self.N, x_target, u_target
        )

    def run_optimization(self, quad_current_state=None, return_x=False):
        """
        Optimizes a trajectory to reach the pre-set target state, starting from the input initial state, that minimizes
        the quadratic cost function and respects the constraints of the system

        :param initial_state: 13-element list of the initial state. If None, 0 state will be used
        :param return_x: bool, whether to also return the optimized sequence of states alongside with the controls.
        :return: optimized control input sequence (flattened)
        """

        return optimize(self.acados_ocp_solver, self.N, quad_current_state)
