""" Implementation of the Simplified Simulator and its quadrotor dynamics.

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


from math import sqrt

import numpy as np
from src.utils.utils import (
    quaternion_inverse,
    quaternion_product,
    unit_quat,
    quaternion_rotate_point,
    fast_cross,
)


class Quadrotor3D:
    def __init__(self, noisy=False, drag=False, payload=False, motor_noise=False):
        """
        Initialization of the 3D quadrotor class
        :param noisy: Whether noise is used in the simulation
        :type noisy: bool
        :param drag: Whether to simulate drag or not.
        :type drag: bool
        :param payload: Whether to simulate a payload force in the simulation
        :type payload: bool
        :param motor_noise: Whether non-gaussian noise is considered in the motor inputs
        :type motor_noise: bool
        """

        # System state space
        self._state = np.empty((10,))
        self.pos = np.zeros((3,))
        self.vel = np.zeros((3,))
        self.angle = np.array([0.0, 0.0, 0.0, 1.0])  # Quaternion format: qw, qx, qy, qz

        self.mass = 1.0  # kg

        # Gravity vector
        self.g = np.array([[0], [0], [9.81]])  # m s^-2

        self.u = np.array([0.0, 0.0, 0.0, 0.0])  # N

        # Drag coefficients [kg / m]
        self.rotor_drag_xy = 0.3
        self.rotor_drag_z = 0.0  # No rotor drag in the z dimension
        self.rotor_drag = np.array(
            [self.rotor_drag_xy, self.rotor_drag_xy, self.rotor_drag_z]
        )[:, np.newaxis]
        self.aero_drag = 0.08

        self.drag = drag
        self.noisy = noisy
        self.motor_noise = motor_noise

        self.payload_mass = 0.3  # kg
        self.payload_mass = self.payload_mass * payload

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, val):
        self._state[:] = np.asarray(val)

    @property
    def pos(self):
        return self._state[0:3]

    @pos.setter
    def pos(self, val):
        self._state[0:3] = np.asarray(val)

    @property
    def angle(self):
        return self._state[3:7]

    @angle.setter
    def angle(self, val):
        self._state[3:7] = np.asarray(val)

    @property
    def vel(self):
        return self._state[7:10]

    @vel.setter
    def vel(self, val):
        self._state[7:10] = np.asarray(val)

    def get_control(self, noisy=False):
        if not noisy:
            return self.u_noiseless
        else:
            return self.u

    def update(self, u, dt):
        """
        Runge-Kutta 4th order dynamics integration

        :param u: 4-dimensional vector with components between [0.0, 1.0] that represent the activation of each motor.
        :param dt: time differential
        """

        self.u[:] = u

        # Generate disturbance forces / torques
        if self.noisy:
            f_d = np.random.normal(size=(3, 1), scale=10 * dt)
        else:
            f_d = np.zeros((3, 1))

        x = self.state

        # RK4 integration
        k1, k2, k3, k4 = (
            np.empty((10,)),
            np.empty((10,)),
            np.empty((10,)),
            np.empty((10,)),
        )
        k1[0:3] = self.f_pos(x)
        k1[3:7] = self.f_att(x, self.u)
        k1[7:10] = self.f_vel(x, self.u, f_d)
        x_aux = x + dt / 2 * k1
        k2[0:3] = self.f_pos(x_aux)
        k2[3:7] = self.f_att(x_aux, self.u)
        k2[7:10] = self.f_vel(x_aux, self.u, f_d)
        x_aux = x + dt / 2 * k2
        k3[0:3] = self.f_pos(x_aux)
        k3[3:7] = self.f_att(x_aux, self.u)
        k3[7:10] = self.f_vel(x_aux, self.u, f_d)
        x_aux = x + dt * k3
        k4[0:3] = self.f_pos(x_aux)
        k4[3:7] = self.f_att(x_aux, self.u)
        k4[7:10] = self.f_vel(x_aux, self.u, f_d)
        x = x + dt * (1.0 / 6.0 * k1 + 2.0 / 6.0 * k2 + 2.0 / 6.0 * k3 + 1.0 / 6.0 * k4)

        # Ensure unit quaternion
        x[3:7] = unit_quat(x[3:7])

        self.pos = x[0:3]
        self.angle = x[3:7]
        self.vel = x[7:10]

    def f_pos(self, x):
        """
        Time-derivative of the position vector
        :param x: 4-length array of input state with components: 3D pos, quaternion angle, 3D vel, 3D rate
        :return: position differential increment (vector): d[pos_x; pos_y]/dt
        """

        vel = quaternion_rotate_point(x[7:10], quaternion_inverse(x[3:7]))
        return vel

    def f_att(self, x, u):
        """
        Time-derivative of the attitude in quaternion form
        :param x: 4-length array of input state with components: 3D pos, quaternion angle, 3D vel, 3D rate
        :return: attitude differential increment (quaternion qw, qx, qy, qz): da/dt
        """

        rate_q = np.zeros((4,))
        rate_q[0:3] = -0.5 * u[1:4]
        angle_quaternion = x[3:7]

        return quaternion_product(rate_q, angle_quaternion)

    def f_vel(self, x, u, f_d):
        """
        Time-derivative of the velocity vector
        :param x: 4-length array of input state with components: 3D pos, quaternion angle, 3D vel, 3D rate
        :param u: control input vector (4-dimensional): [trust_motor_1, ..., thrust_motor_4]
        :param f_d: disturbance force vector (3-dimensional)
        :return: 3D velocity differential increment (vector): d[vel_x; vel_y; vel_z]/dt
        """

        a_thrust = np.array([[0], [0], [u[0]]])

        if self.drag:
            # Transform velocity to body frame
            v_b = x[7:10]
            # Compute aerodynamic drag acceleration in world frame
            a_drag = -self.aero_drag * v_b**2 * np.sign(v_b) / self.mass
            # Add rotor drag
            a_drag -= self.rotor_drag * v_b / self.mass
            # Transform drag acceleration to world frame
            a_drag = quaternion_rotate_point(a_drag, x[3:7])
        else:
            a_drag = np.zeros((3, 1))

        angle_quaternion = x[3:7]

        return np.squeeze(
            -fast_cross(u[1:4], x[7:10])[..., None]
            + a_thrust
            + f_d / self.mass
            + quaternion_rotate_point(-self.g, angle_quaternion)
        )
