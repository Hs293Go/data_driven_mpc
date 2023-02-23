import casadi as cs
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from src.utils.utils import (
    quaternion_inverse,
    quaternion_product,
    quaternion_rotate_point,
)


def make_acados_optimizer(
    t_horizon,
    n_nodes,
    q_cost,
    r_cost,
    q_mask,
    model_name,
    solver_options=None,
    solver_kw=None,
):
    # Weighted squared error loss function q = (p_xyz, a_xyz, v_xyz, r_xyz), r = (u1, u2, u3, u4)
    if q_cost is None:
        q_cost = np.array(
            [10, 10, 10, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        )
    if r_cost is None:
        r_cost = np.array([0.1, 0.1, 0.1, 0.1])

    acados_data = {
        "T": t_horizon,
        "N": n_nodes,
    }
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
    ocp.dims.N = n_nodes
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
    if solver_kw is not None:
        solver = AcadosOcpSolver(ocp, **solver_kw)
    else:
        solver = AcadosOcpSolver(ocp)

    return solver, acados_data


def set_reference_trajectory(acados_ocp_solver, N, x_reference, u_reference):
    if u_reference is not None:
        if not (
            x_reference[0].shape[0] == (u_reference.shape[0] + 1)
            or x_reference[0].shape[0] == u_reference.shape[0]
        ):
            return False

    # If not enough states in target sequence, append last state until required length is met
    while x_reference[0].shape[0] < N + 1:
        x_reference = [
            np.concatenate((x, np.expand_dims(x[-1, :], 0)), 0) for x in x_reference
        ]
        if u_reference is not None:
            u_reference = np.concatenate(
                (u_reference, np.expand_dims(u_reference[-1, :], 0)), 0
            )

    stacked_x_reference = np.concatenate([x for x in x_reference], 1)

    for j in range(N):
        ref = stacked_x_reference[j, :]
        ref = np.concatenate((ref, u_reference[j, :]))
        acados_ocp_solver.set(j, "yref", ref)
    # the last MPC node has only a state reference but no input reference
    acados_ocp_solver.set(N, "yref", stacked_x_reference[N, :])

    return True


def set_reference_state(acados_ocp_solver, N, x_reference, u_reference):
    ref = np.concatenate([x_reference[i] for i in range(3)])
    #  Transform velocity to body frame

    ref = np.concatenate((ref, u_reference))

    for j in range(N):
        acados_ocp_solver.set(j, "yref", ref)
    acados_ocp_solver.set(N, "yref", ref[:-4])

    return True


def optimize(acados_ocp_solver, N, quad_current_state):
    # Set initial state. Add gp state if needed
    x_init = quad_current_state
    x_init = np.stack(x_init)

    # Set initial condition, equality constraint
    acados_ocp_solver.set(0, "lbx", x_init)
    acados_ocp_solver.set(0, "ubx", x_init)

    # Solve OCP
    acados_ocp_solver.solve()

    # Get u
    w_opt_acados = np.ndarray((N, 4))
    x_opt_acados = np.ndarray((N + 1, len(x_init)))
    x_opt_acados[0, :] = acados_ocp_solver.get(0, "x")
    for i in range(N):
        w_opt_acados[i, :] = acados_ocp_solver.get(i, "u")
        x_opt_acados[i + 1, :] = acados_ocp_solver.get(i + 1, "x")

    w_opt_acados = np.reshape(w_opt_acados, (-1))
    return (w_opt_acados, x_opt_acados)
