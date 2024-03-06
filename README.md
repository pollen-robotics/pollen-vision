# Pollen Vision

<div align="center">
 
![demo](assets/pollen_vision_intro.gif)

</div>

This repository contains vision related modules used by Pollen Robotics.

Its purpose is to provide a simple and unified interface to use computer vision models curated for their quality and performance when applied to robotics use cases.

**Right now, we support the following models:**

#### Object detection
- `Owl-Vit` for zero-shot object detection and localization
- `Recognize-Anything` for zero-shot object detection (without localization)

#### Object segmentation
- `Mobile-SAM` for (fast) zero-shot object segmentation

Below is an example of combining `Owl-Vit` and `Mobile-Sam` to detect and segment objects in a point cloud, all live. 
(Note: in this example, there is no temporal or spatial filtering of any kind, we display the raw outputs of the models computed independently on each frame)

https://github.com/pollen-robotics/pollen-vision/assets/6552564/a5285627-9cba-4af5-aafb-6af3d1e6d40c




We also provide wrappers for the Luxonis cameras which we use internally. They allow to easily access the main features that are interesting to our robotics applications (RBG-D, onboard h264 encoding and onboard stereo rectification).

# Installation

```
Note: This package has only been tested on Ubuntu 22.04.
```

Install everything in "production" mode:
```console
pip install -e .[all]
```

OR Install only the modules you want: 
```console
pip install -e .[depthai_wrapper]
pip install -e .[vision]
```

Add "dev" mode dependencies (CI/CD, testing, etc):
```console
pip install -e .[dev]
```

## Vision models specific installation information
To use [RAM](https://github.com/pollen-robotics/recognize-anything), you need to manually [download](https://huggingface.co/xinyu1205/recognize-anything-plus-model/blob/main/ram_plus_swin_large_14m.pth) the checkpoint and place it in `pollen_vision/checkpoints/`

## Luxonis depthai specific information

If this is the first time you use luxonis cameras on this computer, you need to setup the udev rules:
```
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

# Usage
## Vision models wrappers
- Recognize Anything (RAM++) : [README](pollen_vision/pollen_vision/vision_models/object_detection/recognize_anything/README.md)
- Owl-Vit : [README](pollen_vision/pollen_vision/vision_models/object_detection/owl_vit/README.md)
- Mobile Sam : [README](pollen_vision/pollen_vision/vision_models/object_segmentation/mobile_sam/README.md)


## Luxonis depthai wrappers
- SDKWrapper and TeleopWrapper: [README](pollen_vision/pollen_vision/camera_wrappers/depthai/README.md)


# (TODO outdated) Run examples

There are three example scripts in `examples/`:
- `SDKWrapper_example.py` -> returns raw cv2 images, with depth and disparity of so specified
- `two_SDKWrappers_example.py` -> Same, but with two devices connected on the host. They are differentiated by their `mx_id`
- `teleopWrapper_example.py` -> returns h264 packets encoded on board

