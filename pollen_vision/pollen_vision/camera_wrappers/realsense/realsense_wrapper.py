from datetime import timedelta
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt
from pollen_vision.camera_wrappers import CameraWrapper


class RealsenseWrapper(CameraWrapper):  # type: ignore
    def __init__(self) -> None:
        import pyrealsense2 as rs

        self.pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.depth, rs.format.z16, 30)
        other_stream, other_format = rs.stream.color, rs.format.rgb8
        config.enable_stream(other_stream, other_format, 30)

        self.pipeline.start(config)
        profile = self.pipeline.get_active_profile()

        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        intrinsics = depth_profile.get_intrinsics()
        fx = float(intrinsics.fx)  # Focal length of x
        fy = float(intrinsics.fy)  # Focal length of y
        ppx = float(intrinsics.ppx)  # Principle Point Offsey of x (aka. cx)
        ppy = float(intrinsics.ppy)  # Principle Point Offsey of y (aka. cy)
        axs = 0.0  # Axis skew

        self.K = np.array([[fx, axs, ppx], [0.0, fy, ppy], [0.0, 0.0, 1.0]])

        self.pc = rs.pointcloud()

    def get_data(
        self,
    ) -> Tuple[Optional[Dict[str, npt.NDArray[np.uint8]]], Optional[Dict[str, float]], Optional[Dict[str, timedelta]]]:
        success, frames = self.pipeline.try_wait_for_frames(timeout_ms=10)

        data = {}
        latency: Optional[Dict[str, float]] = {}
        ts: Optional[Dict[str, timedelta]] = {}
        if not success:
            return None, None, None

        depth = np.asanyarray(frames.get_depth_frame().get_data())
        color_frame = np.asanyarray(frames.get_color_frame().get_data())

        data["left"] = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
        depth = depth * 0.1
        depth = np.array(depth, dtype=np.float32)
        data["depth"] = depth

        return data, latency, ts

    def get_K(self, left: bool = True) -> npt.NDArray[np.float32]:
        return self.K

    def get_D(self, cam_name: str = "left") -> npt.NDArray[np.float32]:
        return np.zeros((5, 1), dtype=np.float32)

if __name__ == "__main__":
    wrapper = RealsenseWrapper()
    while True:
        data, _, _ = wrapper.get_data()  # type: ignore
        if data is None:
            continue
        left = data["left"]
        depth = data["depth"]
        # print(depth[depth != 0.0])
        cv2.imshow("left", left)
        cv2.imshow("depth", depth)
        cv2.waitKey(1)
