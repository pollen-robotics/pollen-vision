# Coding convention for Maths


## Conventions and Scope

We use the Modern Robotics' convention, but they don't cover everything we need when coding. So here's a proposal. We use the mnemotechnic trick of "frames of the same name are close", so we prefix the frame of reference when naming a point.

T_a_c = T_a_b * T_b_c --> Transform composition. The 2 "b" are close, we feel safe :)

P_a_x = T_a_b * P_b_x --> Application to a point. The 2 "b" are close, we feel safe :)


Names should be ONE word long to avoid confusion. Otherwise this is ambiguous: T_shoulder_camera_wrist, is it a transformation from the wrist to the shoulder_camera, or a transformation from the camera_wrist to the shoulder?

We could use capital letters for 3D and lower case letters for 2D:

    "3D homogeneous transformation matrix" --> A pose or transformation (letter T) is a 4x4 matrix with a 3x3 rotation matrix (letter R) + 3x1 column vector (for translation) + 1x4 column vector (for padding). Here a point is a 3D vector (letter P).
    "2D homogeneous transformation matrix" --> A pose or transformation (letter t) is a 3x3 matrix with a 2x2 rotation matrix (letter r) + 3x1 column vector (for translation) + 1x3 row vector (for padding). Here a point is a 2D vector (letter p).

There are other cases but we wouldn't create a convention for them (like sometimes it's useful to pad the 3x3 rotation matrix into a 4x4 rotation matrix...).

What matters is that someone reading your code doesn't struggle to understand it, so if you encounter a situation that's not covered by these conventions just add a clear comment explaining your stuff and be locally coherent with it.

## Code example

Using Python and numpy arrays. if:
```
T_camera_object == The 4x4 pose of the object in the camera frame
T_torso_camera == The 4x4 pose of the camera in the torso frame
T_torso_object == The 4x4 pose of the object in the torso frame
```
then this code is correct:
```
T_torso_object = T_torso_camera @ T_camera_object
```

We can also define the rotation matrix of the transform:
```
R_torso_object = T_torso_object[:3, :3]
```

The 3D point object in the torso frame:
```
P_torso_object = T_torso_object[:3, 3]
```

Which is the same thing as the coordinates x, y and z of the object in the torso frame:
```
x = T_torso_object[0][3]
y = T_torso_object[1][3]
z = T_torso_object[2][3]
```

## References

This post uses Modern Robotics' conventions 
https://www.mecharithm.com/homogenous-transformation-matrices-configurations-in-robotics/