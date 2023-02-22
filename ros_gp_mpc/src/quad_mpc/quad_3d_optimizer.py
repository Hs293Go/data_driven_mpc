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
import shutil
import sys
from copy import copy

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


def make_acados_optimizer(
    t_horizon=1,
    n_nodes=20,
    q_cost=None,
    r_cost=None,
    q_mask=None,
    model_name="quad_3d_acados_mpc",
    solver_options=None,
):
    # Weighted squared error loss function q = (p_xyz, a_xyz, v_xyz, r_xyz), r = (u1, u2, u3, u4)
    if q_cost is None:
        q_cost = np.array(
            [10, 10, 10, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        )
    if r_cost is None:
        r_cost = np.array([0.1, 0.1, 0.1, 0.1])
    T = t_horizon  # Time horizon
    N = n_nodes  # number of control nodes within horizon
    # Declare model variables
    p = cs.MX.sym("p", 3)  # position
    q = cs.MX.sym("a", 4)  # angle quaternion (wxyz)
    v = cs.MX.sym("v", 3)  # velocity

    # Full state vector (13-dimensional)
    x = cs.vertcat(p, q, v)

    # Control input vector
    f = cs.MX.sym("f")
    r = cs.MX.sym("r", 3)  # angle rate

    u = cs.vertcat(f, r)

    g = cs.vertcat(0.0, 0.0, -9.81)
    a_thrust = cs.vertcat(0.0, 0.0, f)

    p_dynamics = quaternion_rotate_point(v, quaternion_inverse(q))
    q_dynamics = 1 / 2 * quaternion_product(cs.vertcat(-r, 0), q)
    v_dynamics = -cs.cross(r, v, 1) + quaternion_rotate_point(g, q) + a_thrust

    f_expl = cs.vertcat(p_dynamics, q_dynamics, v_dynamics)

    x_dot = cs.MX.sym("x_dot", f_expl.shape)
    # cs.Function("x_dot", [x[:10], u], [x_dot], ["x", "u"], ["x_dot"])
    f_impl = x_dot - f_expl

    # Dynamics model
    acados_model = AcadosModel()
    acados_model.f_expl_expr = f_expl
    acados_model.f_impl_expr = f_impl
    acados_model.x = x
    acados_model.xdot = x_dot
    acados_model.u = u
    acados_model.p = []
    acados_model.name = model_name

    # Add one more weight to the rotation (use quaternion norm weighting in acados)
    q_diagonal = np.concatenate(
        (q_cost[:3], np.mean(q_cost[3:6])[np.newaxis], q_cost[3:])
    )
    if q_mask is not None:
        q_mask = np.concatenate((q_mask[:3], np.zeros(1), q_mask[3:]))
        q_diagonal *= q_mask

    nx = acados_model.x.size()[0]
    nu = acados_model.u.size()[0]
    ny = nx + nu
    n_param = acados_model.p.size()[0] if isinstance(acados_model.p, cs.MX) else 0

    # Create OCP object to formulate the optimization
    ocp = AcadosOcp()
    ocp.model = acados_model
    ocp.dims.N = N
    ocp.solver_options.tf = t_horizon

    # Initialize parameters
    ocp.dims.np = n_param
    ocp.parameter_values = np.zeros(n_param)

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    ocp.cost.W = np.diag(np.concatenate((q_diagonal, r_cost)))
    ocp.cost.W_e = np.diag(q_diagonal)
    terminal_cost = (
        0 if solver_options is None or not solver_options["terminal_cost"] else 1
    )
    ocp.cost.W_e *= terminal_cost

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)
    ocp.cost.Vu = np.zeros((ny, nu))
    ocp.cost.Vu[-4:, -4:] = np.eye(nu)

    ocp.cost.Vx_e = np.eye(nx)

    # Initial reference trajectory (will be overwritten)
    x_ref = np.zeros(nx)
    ocp.cost.yref = np.concatenate((x_ref, np.array([9.81, 0.0, 0.0, 0.0])))
    ocp.cost.yref_e = x_ref

    # Initial state (will be overwritten)
    ocp.constraints.x0 = x_ref

    # Set constraints
    ocp.constraints.lbu = np.r_[0.0, -8.0 * np.ones((3,))]
    ocp.constraints.ubu = np.r_[80, 8.0 * np.ones((3,))]
    ocp.constraints.idxbu = np.r_[0:4]

    # Solver options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.print_level = 0
    ocp.solver_options.nlp_solver_type = (
        "SQP_RTI" if solver_options is None else solver_options["solver_type"]
    )
    return ocp


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
        self.acados_ocp_solver = AcadosOcpSolver(
            make_acados_optimizer(
                t_horizon,
                n_nodes,
                q_cost,
                r_cost,
                q_mask,
                model_name,
                solver_options,
            ),
            json_file=json_file,
        )

    def set_reference_state(self, x_target=None, u_target=None):
        """
        Sets the target state and pre-computes the integration dynamics with cost equations
        :param x_target: 13-dimensional target state (p_xyz, a_wxyz, v_xyz, r_xyz)
        :param u_target: 4-dimensional target control input vector (u_1, u_2, u_3, u_4)
        """

        if x_target is None:
            x_target = [[0, 0, 0], [1, 0, 0, 0], [0, 0, 0]]
        if u_target is None:
            u_target = [self.quad.mass * 9.81, 0, 0, 0]

        ref = np.concatenate([x_target[i] for i in range(3)])
        #  Transform velocity to body frame
        v_b = quaternion_rotate_point(ref[7:10], quaternion_inverse(ref[3:7]))
        ref = np.concatenate((ref[:7], v_b, ref[10:]))

        ref = np.concatenate((ref, u_target))

        for j in range(self.N):
            self.acados_ocp_solver.set(j, "yref", ref)
        self.acados_ocp_solver.set(self.N, "yref", ref[:-4])

        return True

    def set_reference_trajectory(self, x_target, u_target):
        """
        Sets the reference trajectory and pre-computes the cost equations for each point in the reference sequence.
        :param x_target: Nx13-dimensional reference trajectory (p_xyz, angle_wxyz, v_xyz, rate_xyz). It is passed in the
        form of a 4-length list, where the first element is a Nx3 numpy array referring to the position targets, the
        second is a Nx4 array referring to the quaternion, two more Nx3 arrays for the velocity and body rate targets.
        :param u_target: Nx4-dimensional target control input vector (u1, u2, u3, u4)
        """

        if u_target is not None:
            if not (
                x_target[0].shape[0] == (u_target.shape[0] + 1)
                or x_target[0].shape[0] == u_target.shape[0]
            ):
                return False

        # If not enough states in target sequence, append last state until required length is met
        while x_target[0].shape[0] < self.N + 1:
            x_target = [
                np.concatenate((x, np.expand_dims(x[-1, :], 0)), 0) for x in x_target
            ]
            if u_target is not None:
                u_target = np.concatenate(
                    (u_target, np.expand_dims(u_target[-1, :], 0)), 0
                )

        stacked_x_target = np.concatenate([x for x in x_target], 1)

        for j in range(self.N):
            ref = stacked_x_target[j, :]
            ref = np.concatenate((ref, u_target[j, :]))
            self.acados_ocp_solver.set(j, "yref", ref)
        # the last MPC node has only a state reference but no input reference
        self.acados_ocp_solver.set(self.N, "yref", stacked_x_target[self.N, :])

        return True

    def run_optimization(self, initial_state=None, return_x=False):
        """
        Optimizes a trajectory to reach the pre-set target state, starting from the input initial state, that minimizes
        the quadratic cost function and respects the constraints of the system

        :param initial_state: 13-element list of the initial state. If None, 0 state will be used
        :param return_x: bool, whether to also return the optimized sequence of states alongside with the controls.
        :return: optimized control input sequence (flattened)
        """

        if initial_state is None:
            initial_state = [0, 0, 0] + [1, 0, 0, 0] + [0, 0, 0] + [0, 0, 0]

        # Set initial state. Add gp state if needed
        x_init = initial_state
        x_init = np.stack(x_init)

        # Set initial condition, equality constraint
        self.acados_ocp_solver.set(0, "lbx", x_init)
        self.acados_ocp_solver.set(0, "ubx", x_init)

        # Set parameters

        # Solve OCP
        self.acados_ocp_solver.solve()

        # Get u
        w_opt_acados = np.ndarray((self.N, 4))
        x_opt_acados = np.ndarray((self.N + 1, len(x_init)))
        x_opt_acados[0, :] = self.acados_ocp_solver.get(0, "x")
        for i in range(self.N):
            w_opt_acados[i, :] = self.acados_ocp_solver.get(i, "u")
            x_opt_acados[i + 1, :] = self.acados_ocp_solver.get(i + 1, "x")

        w_opt_acados = np.reshape(w_opt_acados, (-1))
        return w_opt_acados if not return_x else (w_opt_acados, x_opt_acados)
