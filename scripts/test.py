import depthai as dai

device = dai.Device()
calib = device.readCalibration()
calib.eepromToJsonFile("read_calib.json")
# print(calib)
print(calib.getCameraExtrinsics(dai.CameraBoardSocket.CAM_D, dai.CameraBoardSocket.CAM_C))
