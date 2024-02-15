import argparse

import cv2
import numpy as np
import numpy.typing as npt
import open3d as o3d

from camera_wrappers.depthai.sdk import SDKWrapper
from camera_wrappers.depthai.utils import get_config_file_path, get_config_files_names

# To run this example, you need to install open3d
# pip install open3d

valid_configs = get_config_files_names()

argParser = argparse.ArgumentParser(description="depth wrapper example")
argParser.add_argument(
    "--config",
    type=str,
    required=True,
    choices=valid_configs,
    help=f"Configutation file name : {valid_configs}",
)
args = argParser.parse_args()


def create_point_cloud_from_rgbd(
    rgb_image: npt.NDArray[np.uint8], depth_image: npt.NDArray[np.uint8], K: npt.NDArray[np.float32]
) -> o3d.geometry.PointCloud:
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb_image),
        o3d.geometry.Image(depth_image),
        convert_rgb_to_intensity=False,
    )

    height, width, _ = rgb_image.shape
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, K)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    return pcd


w = SDKWrapper(get_config_file_path(args.config), compute_depth=True)

vis = o3d.visualization.Visualizer()
vis.create_window()
pcd = o3d.geometry.PointCloud()
ctr = vis.get_view_control()
ctr.change_field_of_view(step=100)  # Max fov seems to be 90 in open3d

set_geometry = False

K = w.cam_config.get_K_left()
while True:
    data, _, _ = w.get_data()

    new_pcd = create_point_cloud_from_rgbd(cv2.cvtColor(data["left"], cv2.COLOR_BGR2RGB), data["depth"], K)
    pcd.points = new_pcd.points
    pcd.colors = new_pcd.colors

    if not set_geometry:
        vis.add_geometry(pcd)
        set_geometry = True

    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    disparity = data["disparity"]
    disparity = (disparity * (255 / w.depth_max_disparity)).astype(np.uint8)
    disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)
    cv2.imshow("disparity", disparity)

    cv2.imshow("depthNode_left", data["depthNode_left"])
    cv2.imshow("left", data["left"])

    key = cv2.waitKey(1)
