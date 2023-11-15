import cv2
from cv2 import aruco

ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
CHARUCO_BOARD = aruco.CharucoBoard(
    (11, 8),
    squareLength=0.022,  # These values don't matter at this point
    markerLength=0.0167,  # These values don't matter at this point
    dictionary=ARUCO_DICT,
)

resolution_factor = 3

# To be printed on a A4 sheet (297mm x 210mm)
img = aruco.Board.generateImage(
    CHARUCO_BOARD, (297 * resolution_factor, 210 * resolution_factor), marginSize=40
)
cv2.imwrite("charuco.png", img)
print("Saved charuco.png")
