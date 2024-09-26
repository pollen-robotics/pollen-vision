import cv2
import numpy as np
from cv2 import aruco
from pollen_vision.camera_wrappers import CameraWrapper, SDKWrapper
from pollen_vision.camera_wrappers.depthai.utils import get_config_file_path

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
MARKER_LENGTH = 0.01558
SQUARE_LENGTH = 0.02075
BOARD_SIZE = (11, 8)
BOARD_WIDTH = BOARD_SIZE[0] * SQUARE_LENGTH
BOARD_HEIGHT = BOARD_SIZE[1] * SQUARE_LENGTH
board = cv2.aruco.CharucoBoard(BOARD_SIZE, SQUARE_LENGTH, MARKER_LENGTH, ARUCO_DICT)
board.setLegacyPattern(True)
charuco_detector = cv2.aruco.CharucoDetector(board)


def get_charuco_board_pose_in_cam(image, K, D):  # type: ignore
    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(image)
    if charuco_corners is None:
        return None, None, None

    obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)

    if len(obj_points) < 4:
        return None, None, None

    flag, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, D)

    T_cam_frame = np.eye(4)
    T_cam_frame[:3, :3], _ = cv2.Rodrigues(rvec.flatten())
    T_cam_frame[:3, 3] = tvec.flatten()
    # T_cam_frame = fv_utils.translateInSelf(T_cam_frame, [w / 2, h / 2, 0])

    return T_cam_frame, marker_corners, marker_ids


class FOVCalculator:
    def __init__(self, cam: CameraWrapper):
        self.cam = cam
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
        self.run()

    def run(self) -> None:
        z, w, h = None, None, None
        while True:
            data, _, _ = self.cam.get_data()

            im = data["left"]
            if w is None:
                h, w = im.shape[:2]

            T_cam_board, corners, ids = get_charuco_board_pose_in_cam(
                im, self.cam.get_K("left"), self.cam.get_D("left")
            )  # type: ignore
            if corners is not None and ids is not None:
                im = aruco.drawDetectedMarkers(im, corners, ids)
                z = T_cam_board[:3, 3][2]

            im = cv2.putText(
                im,
                "Move the charuco board so that if fills the entire field of view. Press Enter when done.",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("im", im)
            key = cv2.waitKey(1)
            if key == 13:  # Enter
                break
        self.compute_fov(z, w, h)  # type: ignore

    def compute_fov(self, z, w, h):  # type: ignore
        print("Computing fov ...")

        # Compute the field of view based on K
        fov_x = np.rad2deg(2 * np.arctan(w / (2 * self.cam.get_K("left")[0, 0])))
        fov_y = np.rad2deg(2 * np.arctan(h / (2 * self.cam.get_K("left")[1, 1])))
        print(f"Theoretical Horizontal FOV: {fov_x:.2f} degrees")
        print(f"Theoretical Vertical FOV: {fov_y:.2f} degrees")

        # Compute the field of view based on data
        fov_x = np.rad2deg(2 * np.arctan((BOARD_WIDTH / 2) / z))
        fov_y = np.rad2deg(2 * np.arctan((BOARD_HEIGHT / 2) / z))
        print(f"Measured Horizontal FOV: {fov_x:.2f} degrees")
        print(f"Measured Vertical FOV: {fov_y:.2f} degrees")


if __name__ == "__main__":
    cam = SDKWrapper(get_config_file_path("CONFIG_IMX296"), fps=30, compute_depth=False, rectify=True)
    # cam = TeleopWrapper(get_config_file_path("CONFIG_IMX296"), fps=30)
    fov_calc = FOVCalculator(cam)
