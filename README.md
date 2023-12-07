# Installation

Install in "production" mode:
```console
pip install -e .
```

Install in "dev" mode:
```console
pip install -e .[dev]
```

If this is the first time you use luxonis stuff on this computer, you need to setup the udev rules:
```
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

# Run examples

There are three example scripts in `src/depthai_wrappers/`:
- cvWrapper_example.py -> returns raw cv2 images 
- teleopWrapper_example.py -> returns h264 packets encoded on board
- depthWrapper_example.py -> returns depth and disparity maps

# Calibration 

Check the [calibration](src/calibration/README.md) page for more details.
