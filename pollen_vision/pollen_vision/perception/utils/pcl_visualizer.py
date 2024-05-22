import numpy as np
import numpy.typing as npt
import open3d as o3d


class PCLVisualizer:
    def __init__(self, K: npt.NDArray[np.float64], name: str = "PCLVisualizer") -> None:
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(name)
        self.pcd = o3d.geometry.PointCloud()

        self.frames = {}  # type: ignore

        self.set_geometry = False

        self.K = K

    def create_point_cloud_from_rgbd(
        self, rgb_image: npt.NDArray[np.uint8], depth_image: npt.NDArray[np.float32]
    ) -> o3d.geometry.PointCloud:
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_image),
            o3d.geometry.Image(depth_image),
            convert_rgb_to_intensity=False,
        )

        height, width, _ = rgb_image.shape
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, self.K)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        return pcd

    def add_frame(self, name: str, pose: npt.NDArray[np.float64] = np.eye(4)) -> None:
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        mesh = mesh.transform(pose)

        self.vis.add_geometry(mesh)

        self.frames[name] = {"mesh": mesh, "pose": pose}

    def update(self, rgb: npt.NDArray[np.uint8], depth: npt.NDArray[np.float32]) -> None:
        new_pcd = self.create_point_cloud_from_rgbd(rgb, depth)
        self.pcd.points = new_pcd.points
        self.pcd.colors = new_pcd.colors

        if not self.set_geometry:
            self.vis.add_geometry(self.pcd)
            self.set_geometry = True

        self.vis.update_geometry(self.pcd)

    def updateFramePose(self, name: str, pose: npt.NDArray[np.float64]) -> None:
        if name not in self.frames:
            print("Mesh not added")
            return

        current_pose = self.frames[name]["pose"]
        self.frames[name]["mesh"].transform(np.linalg.inv(current_pose))
        self.frames[name]["mesh"].transform(pose)
        self.vis.update_geometry(self.frames[name]["mesh"])
        self.frames[name]["pose"] = pose

    def tick(self) -> None:
        self.vis.poll_events()
        self.vis.update_renderer()
