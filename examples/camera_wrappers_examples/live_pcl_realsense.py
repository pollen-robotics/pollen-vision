import argparse

import cv2
import numpy as np
import numpy.typing as npt
import open3d as o3d
from pollen_vision.camera_wrappers import RealsenseWrapper

# To run this example, you need to install open3d
# pip install open3d


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


w = RealsenseWrapper()

vis = o3d.visualization.Visualizer()
vis.create_window()
pcd = o3d.geometry.PointCloud()
ctr = vis.get_view_control()
# ctr.change_field_of_view(step=70)  # Max fov seems to be 90 in open3d

set_geometry = False

K = w.get_K()
while True:
    data, _, _ = w.get_data()
    if data is None:
        continue
    new_pcd = create_point_cloud_from_rgbd(cv2.cvtColor(data["left"], cv2.COLOR_BGR2RGB), data["depth"], K)
    pcd.points = new_pcd.points
    pcd.colors = new_pcd.colors

    if not set_geometry:
        vis.add_geometry(pcd)
        set_geometry = True

    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    cv2.imshow("left", data["left"])
    cv2.imshow("depth", cv2.normalize(data["depth"], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

    key = cv2.waitKey(1)
