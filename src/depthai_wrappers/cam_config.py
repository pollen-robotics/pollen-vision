import json


class CamConfig:
    def __init__(self, cam_config_json, fps, resize):
        self.cam_config_json = cam_config_json
        self.fps = fps

        config = json.load(open(self.cam_config_json, "rb"))
        self.socket_to_name = config["socket_to_name"]
        self.inverted = config["inverted"]
        self.fisheye = config["fisheye"]
        self.mono = config["mono"]
        self.name_to_socket = {v: k for k, v in self.socket_to_name.items()}
        self.sensor_resolution = None
        self.resize_resolution = resize
        self.undstort_maps = {"left": None, "right": None}

    def set_sensor_resolution(self, resolution):
        self.sensor_resolution = resolution

        # Assuming that the resize resolution is the same as the sensor resolution until set otherwise
        if self.resize_resolution is None:
            self.resize_resolution = resolution

    def set_resize_resolution(self, resolution):
        self.resize_resolution = resolution

    def set_undistort_maps(self, mapXL, mapYL, mapXR, mapYR):
        self.undstort_maps["left"] = (mapXL, mapYL)
        self.undstort_maps["right"] = (mapXR, mapYR)

    def to_string(self):
        ret_string = ""
        ret_string += "FPS: {}\n".format(self.fps)
        ret_string += "Sensor resolution: {}\n".format(self.sensor_resolution)
        ret_string += "Resize resolution: {}\n".format(self.resize_resolution)
        ret_string += "Inverted: {}\n".format(self.inverted)
        ret_string += "Fisheye: {}\n".format(self.fisheye)
        ret_string += "Mono: {}\n".format(self.mono)
        ret_string += (
            "Undistort maps are: " + "set"
            if self.undstort_maps["left"] is not None
            else "not set"
        )

        return ret_string
