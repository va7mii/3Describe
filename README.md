# About 3Describe
This projects involves using the Intel Realsense D435 depth camera to create 3D-AR objects that can be touched with hands through Mediapipe's hand-tracking feature. The main Python file is **3d-describe-video.py**. When the Python runs (with the depth camera), a green 3D cube (or prism if you configure the dimentions in the code) appears and rotates. When the 3D cube/prism is touched, it becomes blue and becomes pushed in the direction relative to angle where the cube/prism was touched. Also, press **1 key** to show hand-tracking marks from Mediapipe and **2 key** to hide it.

**-Hand behind cube**
![Screenshot 2023-09-09 011641](https://github.com/comrademan/3Describe/assets/85780191/f5e00067-0fce-43f8-9eff-86bd3ea5b07b)

**-Hand above cube (note Mediapipe's hand-tracking marks are turn on for this one)**
![Screenshot 2023-09-09 011726](https://github.com/comrademan/3Describe/assets/85780191/43d55950-21c0-40f4-b48c-5e2d7b3b3ea9)

**-Hand touching cube from behind**
![Screenshot 2023-09-09 011759](https://github.com/comrademan/3Describe/assets/85780191/099122d4-46fb-42b6-9142-35e82e9eaf7a)

**-Hand touching cube from behind (without hand-tracking marks)**
![Screenshot 2023-09-09 011658](https://github.com/comrademan/3Describe/assets/85780191/deff0af9-b401-4650-b65e-0f1ff256fafe)

# How it works

![IMG_3396](https://github.com/comrademan/3Describe/assets/85780191/f8aa484c-df8a-47aa-ab51-e2ec3fea9b1a)


![IMG_3398](https://github.com/comrademan/3Describe/assets/85780191/2c92dd15-040a-4b80-8ace-d514db14b489)

![IMG_3399](https://github.com/comrademan/3Describe/assets/85780191/5bc4d3dd-a962-493f-a291-38786200b68b)


# References
‘Rs-Ar-Basic’. Intel® RealSense™ Developer Documentation, https://dev.intelrealsense.com/docs/rs-ar-basic. 

Smart Design Technology. ‘Hand Detection in 3D Space’. Medium, 10 Mar. 2022, https://medium.com/@smart-design-techology/hand-detection-in-3d-space-888433a1c1f3.

‘Video Game Math: Collision Detection’. Academy of Interactive Entertainment (AIE), https://aie.edu/articles/video-game-math-collision-detection/.
