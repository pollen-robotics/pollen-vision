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
            resize=(640, 480),
            rectify=True,
            exposure_params=None,
            mx_id=mx_id,
            isp_scale=(2, 3),
        )
        self.cvColorMap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
        self.cvColorMap[0] = [0, 0, 0]

    def get_data(self) -> Tuple[Dict[str, npt.NDArray[np.uint8]], Dict[str, float], Dict[str, timedelta]]:
        # message_group = self.queues["depth_left"].get()
        # frame_left = message_group["left"].getCvFrame()
        # frame_depth = message_group["depth"]

        # return {"left": frame_left, "depth": frame_depth}, None, None

        data, latency, ts = super().get_data()
        for name, pkt in data.items():
            if name != "depth":
                data[name] = pkt.getCvFrame()
            else:
                depth = np.array(pkt.getFrame()).astype(np.float32)
                max_depth = (self.tofConfig.phaseUnwrappingLevel + 1) * 1500  # 100MHz modulation freq.
                data[name] = depth  # / max_depth
                # print(max_depth)
                # max_depth = (self.tof_camConfig.phaseUnwrappingLevel + 1) * 1500  # 100MHz modulation freq.
                # depth = (depth / max_depth) * 255  # .astype(np.uint8)
                # depth_colorized = np.interp(depth, (0, max_depth), (0, 255)).astype(np.uint8)
                # depth_colorized = cv2.applyColorMap(depth_colorized, self.cvColorMap)
                # data[name] = depth_colorized
                # data[name] = np.array(pkt.getFrame()).astype(np.float32)

        return data, latency, ts

    def get_K(self) -> npt.NDArray[np.float32]:
        return super().get_K(left=True)  # type: ignore

    def _create_pipeline(self) -> dai.Pipeline:
        pipeline = self._pipeline_basis()

        self.tof = pipeline.create(dai.node.ToF)
        # self.tof.setNumShaves(1)

        self.tofConfig = self.tof.initialConfig.get()
        self.tofConfig.enableOpticalCorrection = False
        self.tofConfig.enablePhaseShuffleTemporalFilter = True
        self.tofConfig.phaseUnwrappingLevel = 1
        self.tofConfig.phaseUnwrapErrorThreshold = 300
        self.tofConfig.enableFPPNCorrection = False
        # self.tofConfig.median = dai.MedianFilter.KERNEL_7x7

        self.tof.initialConfig.set(self.tofConfig)

        self.tof_cam = pipeline.create(dai.node.Camera)
        self.tof_cam.setFps(self.cam_config.fps)
        tof_socket = get_socket_from_name("tof", self.cam_config.name_to_socket)
        self.tof_cam.setBoardSocket(tof_socket)

        # self.sync = pipeline.create(dai.node.Sync)
        # self.sync.setSyncThreshold(timedelta(seconds=(1 / self.cam_config.fps)))

        # self.align = pipeline.create(dai.node.ImageAlign)

        pipeline = self._create_output_streams(pipeline)

        return self._link_pipeline(pipeline)

    def _create_output_streams(self, pipeline: dai.Pipeline) -> dai.Pipeline:
        pipeline = super()._create_output_streams(pipeline)

        self.xout_tof = pipeline.create(dai.node.XLinkOut)
        self.xout_tof.setStreamName("depth")

        # self.xout_tof_left = pipeline.create(dai.node.XLinkOut)
        # self.xout_tof_left.setStreamName("depth_left")

        return pipeline

    def _link_pipeline(self, pipeline: dai.Pipeline) -> dai.Pipeline:
        self.left.isp.link(self.left_manip.inputImage)
        self.right.isp.link(self.right_manip.inputImage)

        self.tof_cam.raw.link(self.tof.input)
        self.tof.depth.link(self.xout_tof.input)
        # self.tof.depth.link(self.align.input)
        # self.left_manip.out.link(self.align.inputAlignTo)
        # self.align.outputAligned.link(self.xout_tof.input)

        # self.left_manip.out.link(self.sync.inputs["left"])
        # self.align.outputAligned.link(self.sync.inputs["depth"])
        # self.sync.inputs["left"].setBlocking(False)
        # self.left_manip.out.link(self.align.inputAlignTo)
        # self.sync.out.link(self.xout_tof_left.input)

        self.right_manip.out.link(self.xout_right.input)
        self.left_manip.out.link(self.xout_left.input)
        return pipeline
        # self.tof_cam.raw.link(self.tof.input)
        # self.left.isp.link(self.left_manip.inputImage)
        # self.right.isp.link(self.right_manip.inputImage)

        # self.left_manip.out.link(self.xout_left.input)
        # self.right_manip.out.link(self.xout_right.input)

        # self.tof.depth.link(self.xout_tof.input)

        # return pipeline

    def _create_queues(self) -> Dict[str, dai.DataOutputQueue]:
        queues: Dict[str, dai.DataOutputQueue] = super()._create_queues()

        queues["depth"] = self._device.getOutputQueue("depth", maxSize=1, blocking=False)

        return queues


mouse_x, mouse_y = 0, 0


def cv_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    mouse_x, mouse_y = x, y


if __name__ == "__main__":
    t = TOFWrapper(get_config_file_path("CONFIG_IMX296_TOF"), fps=60)

    # P = PCLVisualizer(t.get_K())
    cv2.namedWindow("depth")
    cv2.setMouseCallback("depth", cv_callback)
    while True:
        data, _, _ = t.get_data()
        left = cv2.resize(data["left"], (640, 480))
        right = cv2.resize(data["right"], (640, 480))
        depth = data["depth"]
        # cv2.imshow("left", left)
        # cv2.imshow("right", right)
        print(data["depth"][mouse_y, mouse_x])
        depth = cv2.circle(depth, (mouse_x, mouse_y), 5, (0, 255, 0), -1)
        cv2.imshow("depth", depth)
        # P.update(cv2.cvtColor(left, cv2.COLOR_BGR2RGB), depth)
        # P.tick()
        cv2.waitKey(1)
