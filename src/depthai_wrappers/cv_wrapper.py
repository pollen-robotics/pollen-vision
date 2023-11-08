import depthai as dai

from depthai_wrappers.utils import get_socket_from_name
from depthai_wrappers.wrapper import Wrapper


class CvWrapper(Wrapper):
    def __init__(
        self, cam_config_json, fps, force_usb2=False, resize=None, rectify=False
    ):
        super().__init__(
            cam_config_json, fps, force_usb2=force_usb2, resize=resize, rectify=rectify
        )

    def get_data(self):
        data = {}
        for name, queue in self.queues.items():
            data[name] = queue.get().getCvFrame()
        return data

    def create_pipeline(self):
        pipeline = dai.Pipeline()

        left_socket = get_socket_from_name("left", self.cam_config.name_to_socket)
        right_socket = get_socket_from_name("right", self.cam_config.name_to_socket)

        left = pipeline.createColorCamera()
        left.setBoardSocket(left_socket)
        left.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1440X1080)

        right = pipeline.createColorCamera()
        right.setBoardSocket(right_socket)
        right.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1440X1080)

        if self.cam_config.inverted:
            left.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
            right.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)

        left_manipRectify = self.create_manipRectify(
            pipeline, "left", self.cam_config.sensor_resolution, self.rectify
        )
        right_manipRectify = self.create_manipRectify(
            pipeline, "right", self.cam_config.sensor_resolution, self.rectify
        )

        left_manipRescale = self.create_manipResize(
            pipeline, self.cam_config.resize_resolution
        )
        right_manipRescale = self.create_manipResize(
            pipeline, self.cam_config.resize_resolution
        )

        # Declaring output streams
        xout_left = pipeline.createXLinkOut()
        xout_left.setStreamName("left")

        xout_right = pipeline.createXLinkOut()
        xout_right.setStreamName("right")

        # Linking left
        left.isp.link(left_manipRectify.inputImage)
        left_manipRectify.out.link(left_manipRescale.inputImage)
        left_manipRescale.out.link(xout_left.input)

        # Linking right
        right.isp.link(right_manipRectify.inputImage)
        right_manipRectify.out.link(right_manipRescale.inputImage)
        right_manipRescale.out.link(xout_right.input)

        return pipeline

    def create_queues(self):
        queues = {}
        for name in ["left", "right"]:
            queues[name] = self.device.getOutputQueue(name, maxSize=1, blocking=False)
        return queues
