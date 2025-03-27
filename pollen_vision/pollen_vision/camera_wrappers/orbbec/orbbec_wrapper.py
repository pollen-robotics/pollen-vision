from datetime import timedelta
from typing import Dict, Optional, Tuple, Union

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
        self.D = np.zeros(5).astype(np.float32)
        # fx = self.pipeline.get_camera_param().depth_intrinsic.fx
        # fy = self.pipeline.get_camera_param().depth_intrinsic.fy
        # cx = self.pipeline.get_camera_param().depth_intrinsic.cx
        # cy = self.pipeline.get_camera_param().depth_intrinsic.cy

        # Better !
        cam_params = self.pipeline.get_camera_param()
        fx = cam_params.rgb_intrinsic.fx
        fy = cam_params.rgb_intrinsic.fy
        cx = cam_params.rgb_intrinsic.cx
        cy = cam_params.rgb_intrinsic.cy
        self.K[0, 0] = fx
        self.K[1, 1] = fy
        self.K[0, 2] = cx
        self.K[1, 2] = cy

        self.D[0] = cam_params.rgb_distortion.k1
        self.D[1] = cam_params.rgb_distortion.k2
        self.D[2] = cam_params.rgb_distortion.p1
        self.D[3] = cam_params.rgb_distortion.p2
        self.D[4] = cam_params.rgb_distortion.k3

    def i420_to_bgr(self, frame: npt.NDArray[np.uint8], width: int, height: int) -> npt.NDArray[np.uint8]:
        y = frame[0:height, :]
        u = frame[height : height + height // 4].reshape(height // 2, width // 2)
        v = frame[height + height // 4 :].reshape(height // 2, width // 2)
        yuv_image = cv2.merge([y, u, v])
        bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_I420).astype(np.uint8)

        return bgr_image

    def nv21_to_bgr(self, frame: npt.NDArray[np.uint8], width: int, height: int) -> npt.NDArray[np.uint8]:
        y = frame[0:height, :]
        uv = frame[height : height + height // 2].reshape(height // 2, width)
        yuv_image = cv2.merge([y, uv])
        bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV21).astype(np.uint8)

        return bgr_image

    def nv12_to_bgr(self, frame: npt.NDArray[np.uint8], width: int, height: int) -> npt.NDArray[np.uint8]:
        y = frame[0:height, :]
        uv = frame[height : height + height // 2].reshape(height // 2, width)
        yuv_image = cv2.merge([y, uv])
        bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV12).astype(np.uint8)

        return bgr_image

    def frame_to_bgr_image(self, frame: VideoFrame) -> Optional[npt.NDArray[np.uint8]]:
        width = frame.get_width()
        height = frame.get_height()
        color_format = frame.get_format()
        data = np.asanyarray(frame.get_data())
        image = np.zeros((height, width, 3), dtype=np.uint8)
        if color_format == OBFormat.RGB:
            image = np.resize(data, (height, width, 3))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).astype(np.uint8)
        elif color_format == OBFormat.BGR:
            image = np.resize(data, (height, width, 3))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)
        elif color_format == OBFormat.YUYV:
            image = np.resize(data, (height, width, 2))
            image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV).astype(np.uint8)
        elif color_format == OBFormat.MJPG:
            image = cv2.imdecode(data, cv2.IMREAD_COLOR).astype(np.uint8)
        elif color_format == OBFormat.I420:
            image = self.i420_to_bgr(data, width, height)
        elif color_format == OBFormat.NV12:
            image = self.nv12_to_bgr(data, width, height)
        elif color_format == OBFormat.NV21:
            image = self.nv21_to_bgr(data, width, height)
        elif color_format == OBFormat.UYVY:
            image = np.resize(data, (height, width, 2))
            image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY).astype(np.uint8)
        else:
            print("Unsupported color format: {}".format(color_format))
            return None
        return image

    def get_data(
        self,
    ) -> Tuple[
        Dict[str, Union[npt.NDArray[np.uint8], npt.NDArray[np.uint16]]],
        Optional[Dict[str, float]],
        Optional[Dict[str, timedelta]],
    ]:
        data: Dict[str, Union[npt.NDArray[np.uint8], npt.NDArray[np.uint16]]] = {}  # 8 for color 16 for depth
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
        if color_image is None:
            return data, None, None
        color_image = cv2.resize(color_image, (width, height)).astype(np.uint8)

        data["depth"] = depth_data
        data["left"] = color_image
        return data, None, None

    def get_K(self) -> npt.NDArray[np.float32]:
        return self.K

    def get_D(self) -> npt.NDArray[np.float32]:
        return self.D


mouse_x, mouse_y = 0, 0

if __name__ == "__main__":
    o = OrbbecWrapper()
    import time

    def colorizeDepth(frameDepth: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
        invalidMask = frameDepth == 0
        # Log the depth, minDepth and maxDepth
        try:
            minDepth = np.percentile(frameDepth[frameDepth != 0], 3)
            maxDepth = np.percentile(frameDepth[frameDepth != 0], 95)
            logDepth = np.log(frameDepth, where=frameDepth != 0)
            logMinDepth = np.log(minDepth)
            logMaxDepth = np.log(maxDepth)
            np.nan_to_num(logDepth, copy=False, nan=logMinDepth)
            # Clip the values to be in the 0-255 range
            logDepth = np.clip(logDepth, logMinDepth, logMaxDepth)

            # Interpolate only valid logDepth values, setting the rest based on the mask
            depthFrameColor = np.interp(logDepth, (logMinDepth, logMaxDepth), (0, 255))
            depthFrameColor = np.nan_to_num(depthFrameColor)
            depthFrameColor = depthFrameColor.astype(np.uint8)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET).astype(np.uint8)
            # Set invalid depth pixels to black
            depthFrameColor[invalidMask] = 0
        except IndexError:
            # Frame is likely empty
            depthFrameColor = np.zeros((frameDepth.shape[0], frameDepth.shape[1], 3), dtype=np.uint8)
        except Exception as e:
            raise e
        return depthFrameColor.astype(np.uint8)

    # cv2.namedWindow("depth")
    # cv2.setMouseCallback("depth", cv2_callback)

    while True:
        data, _, _ = o.get_data()  # type: ignore
        if "depth" in data:
            cv2.imshow("depth", colorizeDepth(data["depth"].astype(np.uint16)))
            depth_value = data["depth"][mouse_y, mouse_x]
            print(depth_value)
        if "left" in data:
            cv2.imshow("left", data["left"])

        cv2.waitKey(1)
        time.sleep(0.01)
