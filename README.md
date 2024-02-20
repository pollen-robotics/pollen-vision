# Installation

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

## Vision models specific info
To use [RAM](https://github.com/pollen-robotics/recognize-anything), you need to manually [download](https://huggingface.co/xinyu1205/recognize-anything-plus-model/blob/main/ram_plus_swin_large_14m.pth) the checkpoint and place it in `pollen_vision/src/checkpoints/`

## Luxonis depthai specific info

If this is the first time you use luxonis stuff on this computer, you need to setup the udev rules:
```
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

# Usage
## Vision models wrappers
- recognize anything (RAM++) : [README](pollen_vision/pollen_vision/vision_models/object_detection/recognize_anything/README.md)
- owl vit : [README](pollen_vision/pollen_vision/vision_models/object_detection/owl_vit/README.md)
- mobile sam : [README](pollen_vision/pollen_vision/vision_models/object_segmentation/mobile_sam/README.md)


## Luxonis depthai wrappers
- SDKWrapper and TeleopWrapper: [README](pollen_vision/pollen_vision/camera_wrappers/depthai/README.md)


# [DEPRECATED] Run examples

There are three example scripts in `examples/`:
- `SDKWrapper_example.py` -> returns raw cv2 images, with depth and disparity of so specified
- `two_SDKWrappers_example.py` -> Same, but with two devices connected on the host. They are differentiated by their `mx_id`
- `teleopWrapper_example.py` -> returns h264 packets encoded on board

