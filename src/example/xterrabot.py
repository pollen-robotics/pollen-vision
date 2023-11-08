import logging

import numpy as np
import numpy.typing as npt


class XTerraBot:
    """Illustrate Maths notation based on
    https://www.mecharithm.com/
    homogenous-transformation-matrices-configurations-in-robotics/."""

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)
        # b is the mobile base
        # d is the camera
        self._T_d_b = np.array(
            [[0, 0, -1, 250], [0, -1, 0, -150], [-1, 0, 0, 200], [0, 0, 0, 1]]
        )

        self._T_d_e = np.array(
            [[0, 0, -1, 300], [0, -1, 0, 100], [-1, 0, 0, 120], [0, 0, 0, 1]]
        )  # e is the object
        self._T_b_c = np.array(
            [
                [0, -1 / np.sqrt(2), -1 / np.sqrt(2), 30],
                [0, 1 / np.sqrt(2), -1 / np.sqrt(2), -40],
                [1, 0, 0, 25],
                [0, 0, 0, 1],
            ]
        )  # c is the joint of the gripper
        self._T_a_d = np.array(
            [[0, 0, -1, 400], [0, -1, 0, 50], [-1, 0, 0, 300], [0, 0, 0, 1]]
        )  # a is the root

    def get_object_in_gripper_frame(self) -> npt.NDArray[np.float64]:
        T_c_e = (
            np.linalg.inv(self._T_b_c)
            @ np.linalg.inv(self._T_d_b)
            # @ np.linalg.inv(self._T_a_d)
            # @ self._T_a_d ## inv(self._T_a_d) @ self._T_a_d = 1
            @ self._T_d_e
        )
        return np.array(T_c_e)
