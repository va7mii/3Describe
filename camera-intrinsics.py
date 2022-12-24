import pyrealsense2 as rs2
pipe = rs2.pipeline()
selection = pipe.start()
depth_stream = selection.get_stream(rs2.stream.depth).as_video_stream_profile()
resolution = (depth_stream.width(), depth_stream.height())
print(resolution)

i = depth_stream.get_intrinsics()
principal_point = (i.ppx, i.ppy)
focal_length = (i.fx, i.fy)
model = i.model

print(i)
print(principal_point)
print(focal_length)

