"""Fisheye Undistortion

This script reads a JSON file containing camera calibration data and undistorts a fisheye image, using
equirectangular or planar projection.

Raw data can be aquired from the script export_calib.py.

Example usage:

python pollen_vision/pollen_vision/camera_wrappers/depthai/calibration/fisheye_undistort.py \
    calibration_20250112_122016.json right_raw_20250112_122031.jpg \
    right_raw_20250112_122031_undistorted.jpg \
    --projection_type planar \
    --output_size 1440 1080 \
    --side right

python pollen_vision/pollen_vision/camera_wrappers/depthai/calibration/fisheye_undistort.py \
     calibration_20250112_122016.json left_raw_20250112_122031.jpg \
     left_raw_20250112_122031_undistorted.jpg \
     --projection_type equirectangular \
     --output_size 1440 1440
"""

import argparse
import json
import logging

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def read_json_file(file_path):
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from file: {file_path}")
        raise


def read_config(file_path, side: str):
    try:
        data = read_json_file(file_path)

        if side == "left":
            cam_id = data["stereoRectificationData"]["leftCameraSocket"]
            R = np.array(data["stereoRectificationData"]["rectifiedRotationLeft"])
        else:
            cam_id = data["stereoRectificationData"]["rightCameraSocket"]
            R = np.array(data["stereoRectificationData"]["rectifiedRotationRight"])

        if data["cameraData"][0][0] == cam_id:
            cam_id = 0
        else:
            cam_id = 1

        # there is 4 distortion coefficients with the fisheye model
        D = np.array(data["cameraData"][cam_id][1]["distortionCoeff"])[:4]
        K = np.array(data["cameraData"][cam_id][1]["intrinsicMatrix"])

        if side == "left":
            T = np.zeros(3)
        else:
            # rectify vertical translation
            T = np.array(
                [
                    0,  # data["cameraData"][cam_id][1]["extrinsics"]["translation"]["x"],
                    data["cameraData"][cam_id][1]["extrinsics"]["translation"]["y"],
                    0,  # data["cameraData"][cam_id][1]["extrinsics"]["translation"]["z"],
                ]
            )

        logging.info(f"Distortion Coefficients: {D}")
        logging.info(f"Intrinsic Matrix: {K}")
        logging.info(f"Rectified Rotation Matrix: {R}")
        logging.info(f"Translation Vector: {T}")

        return K, D, R, T

    except KeyError as e:
        logging.error(f"Key error: {e}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

    return None, None, None, None


def rectify(x, y, z, R, T):
    X = np.array([x, y, z])
    X = np.dot(R, X) + T
    return X[0], X[1], X[2]


def compute_3D_point_sphere(x, y, half_size_fisheye, size_fisheye, resolution):
    theta = half_size_fisheye[0] - x * size_fisheye[0] / resolution[0]
    phi = half_size_fisheye[1] - y * size_fisheye[1] / resolution[1]

    # opencv model cannot capture larger FoV than 180 degrees
    if theta > np.pi / 2:
        return 0, 0, -1
    elif theta < -np.pi / 2:
        return 0, 0, -1

    # rho = 1
    z_3d = np.cos(phi) * np.cos(theta)  # * rho
    x_3d = -np.cos(phi) * np.sin(theta)  # * rho
    y_3d = -np.sin(phi)  # * rho

    return x_3d, y_3d, z_3d


def compute_3D_point_plan(x, y, half_size_fisheye, size_fisheye, resolution):
    theta = half_size_fisheye[0] - x * size_fisheye[0] / resolution[0]
    phi = half_size_fisheye[1] - y * size_fisheye[1] / resolution[1]

    z_3d = 1  # rho
    x_3d = -np.tan(theta)  # * rho
    y_3d = -np.tan(phi)  # * rho

    return x_3d, y_3d, z_3d


def compute_u_v(x_3d, y_3d, z_3d, K, D, R, T):

    if z_3d == 0:
        z_3d = np.finfo(float).eps
    elif z_3d == -1:
        return 0, 0

    x_3d, y_3d, z_3d = rectify(x_3d, y_3d, z_3d, R, T)

    a = x_3d / z_3d
    b = y_3d / z_3d
    r_square = a**2 + b**2
    r = np.sqrt(r_square)

    theta_fisheye = np.arctan(r)
    theta_fisheye_d = (
        theta_fisheye
        + D[0] * (theta_fisheye**3)
        + D[1] * (theta_fisheye**5)
        + D[2] * (theta_fisheye**7)
        + D[3] * (theta_fisheye**9)
    )

    if r == 0:
        x_p = 0
        y_p = 0
    else:
        x_p = (theta_fisheye_d / r) * a
        y_p = (theta_fisheye_d / r) * b

    u = (int)(K[0][0] * x_p + K[0][2])
    v = (int)(K[1][1] * y_p + K[1][2])

    return u, v


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read and process a JSON file.")
    parser.add_argument("config_path", type=str, help="Path to the JSON file")
    parser.add_argument("input_image_path", type=str, help="Path to the input image")
    parser.add_argument("output_image_path", type=str, help="Path to save the output image")
    parser.add_argument(
        "--output_size",
        type=int,
        nargs=2,
        default=(1440, 1080),
        help="Output image size as width and height (default: 1440 1080)",
    )
    parser.add_argument(
        "--projection_type",
        type=str,
        choices=["equirectangular", "planar"],
        default="planar",
        help="Projection type (default: planar)",
    )
    parser.add_argument(
        "--side",
        type=str,
        choices=["left", "right"],
        default="left",
        help="Select left or right camera parameters",
    )
    args = parser.parse_args()

    logging.info(
        (
            f"File path: {args.config_path}, "
            f"Input image path: {args.input_image_path}, "
            f"Output image path: {args.output_image_path}, "
            f"Output size: {args.output_size}, "
            f"Projection type: {args.projection_type}"
            f"Side: {args.side}"
        )
    )

    K, D, R, T = read_config(args.config_path, args.side)
    undistorded_image = np.zeros((args.output_size[1], args.output_size[0], 3), np.uint8)

    img_distorded = cv2.imread(args.input_image_path)

    if args.projection_type == "equirectangular":
        size_fisheye = np.array([2 * np.pi, np.pi])
    else:
        size_fisheye = np.array([np.pi / 1.65, np.pi / 2])
    half_size_fisheye = np.array([size_fisheye[0] / 2, size_fisheye[1] / 2])
    resolution = np.array([img_distorded.shape[0], img_distorded.shape[1]])
    resolution_undistorted = np.array([args.output_size[0], args.output_size[1]])

    logging.info("Start undistorting the image ...")
    for x in range(resolution_undistorted[0]):
        for y in range(resolution_undistorted[1]):

            if args.projection_type == "equirectangular":
                x_3d, y_3d, z_3d = compute_3D_point_sphere(x, y, half_size_fisheye, size_fisheye, resolution_undistorted)
            else:
                x_3d, y_3d, z_3d = compute_3D_point_plan(x, y, half_size_fisheye, size_fisheye, resolution_undistorted)

            u, v = compute_u_v(x_3d, y_3d, z_3d, K, D, R, T)

            if v >= resolution[0] or v < 0:
                undistorded_image[y, x] = (0, 0, 0)
            elif u >= resolution[1] or u < 0:
                undistorded_image[y, x] = (0, 0, 0)
            else:
                undistorded_image[y, x] = img_distorded[v, u]

    logging.info("Undistortion completed. Writing image to file.")
    cv2.imwrite(args.output_image_path, undistorded_image)
