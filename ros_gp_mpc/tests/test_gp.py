from math import isclose
from pathlib import Path
import pytest
import numpy as np
import numpy.testing as npt
import casadi as cs
from src.model_fitting.internal.kernel_function import CustomKernelFunctions
from src.model_fitting.gp import CustomGPRegression, GPEnsemble
from test_functions import x_by_sin_x, rosenbrock

data_file = Path(__file__).parent.resolve() / "test_data.npz"
np.set_printoptions(precision=16)


def test_1d_fitting():
    rng = np.random.default_rng(42)
    x_test = np.linspace(0, 10, 1000)[:, None]
    x_train = rng.choice(x_test, 6)
    y_train = x_by_sin_x(x_train)
    gp = CustomGPRegression(
        x_features=[0],
        u_features=[0],
        reg_dim=0,
        y_mean=np.mean(y_train),
        n_restarts=12,
        sigma_n=1e-8,
        seed=114514,
    )

    gp.fit(x_train, y_train)
    result = gp.kernel.theta
    expected = np.array([1.2200324282991304, 10.000000000000002])
    npt.assert_almost_equal(result, expected)

    result = np.squeeze(gp.predict(x_test))
    result = result[::100]
    expected = np.array(
        [
            0.8430401107327807,
            0.6969753621345318,
            0.7879104797764924,
            -0.1801491306503742,
            -3.076482447935014,
            -4.863480117882105,
            -1.795528409948922,
            4.791591650990696,
            7.7669454882918085,
            4.711195843680146,
        ]
    )
    npt.assert_almost_equal(result, expected)

    x_test = cs.MX(x_test)
    result = [
        float(cs.evalf(gp.predict(x_test[::100][idx])["mu"]))
        for idx in range(expected.size)
    ]

    npt.assert_almost_equal(result, expected)


def test_2d_fitting():
    rng = np.random.default_rng(42)

    interval = np.linspace(-0.5, 0.5, 100)
    xv, yv = np.meshgrid(interval, interval)
    x_test = np.column_stack((xv.ravel(), yv.ravel()))
    n_samples = 250
    x_train = rng.choice(x_test, n_samples)
    z_train = np.array(list(map(rosenbrock, x_train)))

    gp = CustomGPRegression(
        x_features=np.r_[0:2],
        u_features=np.r_[0:2],
        reg_dim=np.r_[0:2],
        y_mean=np.mean(z_train),
        n_restarts=12,
        sigma_n=1e-8,
        kernel=CustomKernelFunctions(params=dict(l=np.ones((2,)), sigma_f=1)),
        seed=114514,
    )
    gp.fit(x_train, z_train)
    result = gp.kernel.theta
    expected = np.array([0.15064832073180132, 0.05910259657083655, 10.000000000000002])
    npt.assert_almost_equal(result, expected)
    result = np.squeeze(gp.predict(x_test))
    result = result[::1000]
    expected = np.array(
        [
            45.0396404987228,
            44.51756787462873,
            27.914275425817443,
            21.355082779718852,
            13.855567704133692,
            8.140330864880646,
            3.960678121705895,
            2.428077064938382,
            2.7652990987637622,
            5.808993606404267,
        ]
    )
    npt.assert_almost_equal(result, expected)
