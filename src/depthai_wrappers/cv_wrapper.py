from depthai_wrappers.wrapper import Wrapper


class CvWrapper(Wrapper):
    def __init__(self, cam_config_json, fps, force_usb2=False):
        super().__init__(cam_config_json, fps, force_usb2=force_usb2)

    def create_pipeline(self):
        pass
