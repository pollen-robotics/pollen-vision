# left -> right
# left -> tof
# tof -> left

ch.setCameraExtrinsics(
    left_socket,
    tof_socket,
    R_left_to_tof,
    T_left_to_tof,
    specTranslation=T_left_to_tof,
)

ch.setCameraExtrinsics(
    right_socket,
    left_socket,
    R_right_to_left,
    T_right_to_left,
    specTranslation=T_right_to_left,
)

# Should I do this ?
ch.setCameraExtrinsics(
    tof_socket,
    tof_socket,
    [0, 0, 0],
    np.eye(3).tolist(),
    specTranslation=[0, 0, 0],
)
