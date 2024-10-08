from datetime import timedelta
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
from pollen_vision.camera_wrappers import CameraWrapper
from pyorbbecsdk import (
    AlignFilter,
    Config,
    OBFormat,
    OBSensorType,
    OBStreamType,
    Pipeline,
    VideoFrame,
)

MIN_DEPTH = 200  # 20mm
MAX_DEPTH = 50000  # 10000mm


class OrbbecWrapper(CameraWrapper):  # type: ignore
    def __init__(self) -> None:
        self.config = Config()
        self.pipeline = Pipeline()
        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = profile_list.get_default_video_stream_profile()
            self.config.enable_stream(color_profile)
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_profile = profile_list.get_default_video_stream_profile()
            self.config.enable_stream(depth_profile)
            self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        except Exception as e:
            print(e)
            return

        try:
            self.pipeline.enable_frame_sync()
        except Exception as e:
            print(e)

        try:
            self.pipeline.start(self.config)
        except Exception as e:
            print(e)
            return

        self.K = np.eye(3).astype(np.float32)
        # fx = self.pipeline.get_camera_param().depth_intrinsic.fx
        # fy = self.pipeline.get_camera_param().depth_intrinsic.fy
        # cx = self.pipeline.get_camera_param().depth_intrinsic.cx
        # cy = self.pipeline.get_camera_param().depth_intrinsic.cy

        # Better !
        fx = self.pipeline.get_camera_param().rgb_intrinsic.fx
        fy = self.pipeline.get_camera_param().rgb_intrinsic.fy
        cx = self.pipeline.get_camera_param().rgb_intrinsic.cx
        cy = self.pipeline.get_camera_param().rgb_intrinsic.cy
        self.K[0, 0] = fx
        self.K[1, 1] = fy
        self.K[0, 2] = cx
        self.K[1, 2] = cy

    def i420_to_bgr(self, frame: npt.NDArray, width: int, height: int) -> npt.NDArray[np.uint8]:  # type: ignore
        y = frame[0:height, :]
        u = frame[height : height + height // 4].reshape(height // 2, width // 2)
        v = frame[height + height // 4 :].reshape(height // 2, width // 2)
        yuv_image = cv2.merge([y, u, v])
        bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_I420)

        return bgr_image  # type: ignore

    def nv21_to_bgr(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:  # type: ignore
        y = frame[0:height, :]
        uv = frame[height : height + height // 2].reshape(height // 2, width)
        yuv_image = cv2.merge([y, uv])
        bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV21)

        return bgr_image  # type: ignore

    def nv12_to_bgr(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:  # type: ignore
        y = frame[0:height, :]
        uv = frame[height : height + height // 2].reshape(height // 2, width)
        yuv_image = cv2.merge([y, uv])
        bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV12)

        return bgr_image  # type: ignore

    def frame_to_bgr_image(self, frame: VideoFrame) -> Union[Optional[np.array], Any]:  # type: ignore
        width = frame.get_width()
        height = frame.get_height()
        color_format = frame.get_format()
        data = np.asanyarray(frame.get_data())
        image = np.zeros((height, width, 3), dtype=np.uint8)
        if color_format == OBFormat.RGB:
            image = np.resize(data, (height, width, 3))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif color_format == OBFormat.BGR:
            image = np.resize(data, (height, width, 3))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif color_format == OBFormat.YUYV:
            image = np.resize(data, (height, width, 2))
            image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
        elif color_format == OBFormat.MJPG:
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        elif color_format == OBFormat.I420:
            image = self.i420_to_bgr(data, width, height)
            return image
        elif color_format == OBFormat.NV12:
            image = self.nv12_to_bgr(data, width, height)
            return image
        elif color_format == OBFormat.NV21:
            image = self.nv21_to_bgr(data, width, height)
            return image
        elif color_format == OBFormat.UYVY:
            image = np.resize(data, (height, width, 2))
            image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
        else:
            print("Unsupported color format: {}".format(color_format))
            return None
        return image

    def get_data(self) -> Tuple[Dict[str, npt.NDArray[np.uint8]], Optional[Dict[str, float]], Optional[Dict[str, timedelta]]]:
        data: Dict[str, npt.NDArray[np.uint8]] = {}
        frames = self.pipeline.wait_for_frames(100)
        if not frames:
            return data, None, None
        frames = self.align_filter.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if depth_frame is None or color_frame is None:
            return data, None, None
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        scale = depth_frame.get_depth_scale()

        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        depth_data = depth_data.reshape((height, width))

        depth_data = depth_data.astype(np.float32) * scale
        depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)
        depth_data = depth_data.astype(np.uint16)

        color_image = self.frame_to_bgr_image(color_frame)
        color_image = cv2.resize(color_image, (width, height))

        data["depth"] = depth_data  # type: ignore
        data["left"] = color_image
        return data, None, None

    def get_K(self) -> npt.NDArray[np.float32]:
        return self.K


mouse_x, mouse_y = 0, 0


# def cv2_callback(event, x, y, flags, param):
#     global mouse_x, mouse_y
#     mouse_x, mouse_y = x, y


if __name__ == "__main__":
    o = OrbbecWrapper()

    # cv2.namedWindow("depth")
    # cv2.setMouseCallback("depth", cv2_callback)

    while True:
        data, _, _ = o.get_data()
        # if "depth" in data:
        #     cv2.imshow("depth", data["depth"])
        #     depth_value = data["depth"][mouse_y, mouse_x]
        #     print(depth_value)
        if "left" in data:
            cv2.imshow("left", data["left"])

        cv2.waitKey(1)
        # time.sleep(0.01)
