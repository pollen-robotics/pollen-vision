from datetime import timedelta
from typing import Dict, Tuple

import cv2
import depthai as dai
import numpy as np
import numpy.typing as npt
from pollen_vision.camera_wrappers.depthai.utils import (
    get_config_file_path,
    get_socket_from_name,
)
from pollen_vision.camera_wrappers.depthai.wrapper import DepthaiWrapper

# from pollen_vision.perception.utils.pcl_visualizer import PCLVisualizer


class TOFWrapper(DepthaiWrapper):
    def __init__(
        self,
        cam_config_json: str,
        fps: int = 30,
        force_usb2: bool = False,
        mx_id: str = "",
    ) -> None:
        super().__init__(
            cam_config_json,
            fps,
            force_usb2=force_usb2,
            resize=None,
            rectify=False,
            exposure_params=None,
            mx_id=mx_id,
            # isp_scale=(2, 3),
        )
        self.cvColorMap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
        self.cvColorMap[0] = [0, 0, 0]
        self.cam_config.undistort_resolution = (640, 480)

    def crop_image(self, im, depth):
        # threshold depth
        thresholded_depth = np.where(depth > 0, 1, 0).astype(np.uint8)
        depth_mask_bounding_box = cv2.boundingRect(thresholded_depth)
        cropped_image = im[
            depth_mask_bounding_box[1] : depth_mask_bounding_box[1] + depth_mask_bounding_box[3],
            depth_mask_bounding_box[0] : depth_mask_bounding_box[0] + depth_mask_bounding_box[2],
        ]
        cropped_depth = depth[
            depth_mask_bounding_box[1] : depth_mask_bounding_box[1] + depth_mask_bounding_box[3],
            depth_mask_bounding_box[0] : depth_mask_bounding_box[0] + depth_mask_bounding_box[2],
        ]

        cropped_image = cv2.resize(cropped_image, (640, 480))
        cropped_depth = cv2.resize(cropped_depth, (640, 480))

        return cropped_image, cropped_depth

    def get_data(self) -> Tuple[Dict[str, npt.NDArray[np.uint8]], Dict[str, float], Dict[str, timedelta]]:
        data: Dict[str, npt.NDArray[np.uint8]] = {}
        latency: Dict[str, float] = {}
        ts: Dict[str, timedelta] = {}

        messageGroup = self.queues["sync_out"].get()

        left = messageGroup["left"].getCvFrame()
        right = messageGroup["right"].getCvFrame()
        depth = messageGroup["depth_aligned"].getFrame()

        # Temporary, not ideal
        cropped_left, cropped_depth = self.crop_image(left, depth)

        data["left"] = cropped_left
        data["right"] = right
        data["depth"] = cropped_depth

        # data["left"] = left
        # data["right"] = right
        # data["depth"] = depth

        return data, latency, ts

    def get_K(self) -> npt.NDArray[np.float32]:
        return super().get_K(left=True)  # type: ignore

    def _create_pipeline(self) -> dai.Pipeline:
        pipeline = self._pipeline_basis()

        self.cam_tof = pipeline.create(dai.node.Camera)
        self.cam_tof.setFps(self.cam_config.fps)
        self.tof_socket = get_socket_from_name("tof", self.cam_config.name_to_socket)
        self.cam_tof.setBoardSocket(self.tof_socket)

        self.tof = pipeline.create(dai.node.ToF)

        # === Tof configuration ===
        self.tofConfig = self.tof.initialConfig.get()
        self.tofConfig.enableFPPNCorrection = True
        self.tofConfig.enableOpticalCorrection = False  # True is ok ?
        self.tofConfig.enablePhaseShuffleTemporalFilter = True
        self.tofConfig.phaseUnwrappingLevel = 1
        self.tofConfig.phaseUnwrapErrorThreshold = 300
        self.tofConfig.enableTemperatureCorrection = False  # Not yet supported
        self.tofConfig.enableWiggleCorrection = False
        self.tofConfig.median = dai.MedianFilter.KERNEL_7x7
        self.tof.initialConfig.set(self.tofConfig)
        # ==========================

        self.align = pipeline.create(dai.node.ImageAlign)

        self.sync = pipeline.create(dai.node.Sync)
        self.sync.setSyncThreshold(timedelta(seconds=(0.5 / self.cam_config.fps)))

        pipeline = self._create_output_streams(pipeline)

        return self._link_pipeline(pipeline)

    def _create_output_streams(self, pipeline: dai.Pipeline) -> dai.Pipeline:
        pipeline = super()._create_output_streams(pipeline)

        self.xout_sync = pipeline.create(dai.node.XLinkOut)
        self.xout_sync.setStreamName("sync_out")

        return pipeline

    def _link_pipeline(self, pipeline: dai.Pipeline) -> dai.Pipeline:
        self.left.isp.link(self.left_manip.inputImage)
        self.right.isp.link(self.right_manip.inputImage)

        self.left_manip.out.link(self.sync.inputs["left"])
        self.right_manip.out.link(self.sync.inputs["right"])
        self.sync.inputs["left"].setBlocking(False)
        self.sync.inputs["right"].setBlocking(False)

        self.cam_tof.raw.link(self.tof.input)

        # Align depth on RGB WORKS
        self.left_manip.out.link(self.align.inputAlignTo)
        self.tof.depth.link(self.align.input)

        # Align RGB on depth
        # self.left_manip.out.link(self.align.input)
        # self.tof.depth.link(self.align.inputAlignTo)

        self.align.outputAligned.link(self.sync.inputs["depth_aligned"])

        self.sync.out.link(self.xout_sync.input)

        return pipeline

    def _create_queues(self) -> Dict[str, dai.DataOutputQueue]:
        # queues: Dict[str, dai.DataOutputQueue] = super()._create_queues()

        queues: Dict[str, dai.DataOutputQueue] = {}
        queues["sync_out"] = self._device.getOutputQueue("sync_out", maxSize=8, blocking=False)

        return queues


mouse_x, mouse_y = 0, 0


def cv_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    mouse_x, mouse_y = x, y


if __name__ == "__main__":
    t = TOFWrapper(get_config_file_path("CONFIG_AR0234_TOF"), fps=30)

    # P = PCLVisualizer(t.get_K())
    cv2.namedWindow("depth")
    cv2.setMouseCallback("depth", cv_callback)
    print(dai.__version__)
    while True:
        data, _, _ = t.get_data()
        left = data["left"]
        right = data["right"]
        # left = cv2.resize(left, (640, 480))
        # right = cv2.resize(right, (640, 480))
        depth = data["depth"]
        cv2.imshow("left", left)
        cv2.imshow("right", right)
        print(data["depth"][mouse_y, mouse_x])
        depth = cv2.circle(depth, (mouse_x, mouse_y), 5, (0, 255, 0), -1)
        cv2.imshow("depth", depth)
        # P.update(cv2.cvtColor(left, cv2.COLOR_BGR2RGB), depth)
        # P.tick()
        cv2.waitKey(1)
