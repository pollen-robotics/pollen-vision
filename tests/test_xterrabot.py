import numpy as np
import pytest

from src.example.xterrabot import XTerraBot


def test_xterrabot():
    xtbot = XTerraBot()
    T_c_e = xtbot.get_object_in_gripper_frame()
    T_c_e_ref = np.array(
        [
            [0, 0, 1, -75],
            [-0.7071, 0.7071, 0, -183.8478],
            [-0.7071, -0.7071, 0, 113.1371],
            [0, 0, 0, 1],
        ]
    )

    assert np.allclose(T_c_e, T_c_e_ref)
