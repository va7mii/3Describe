#!/usr/bin/env python
"""
Quick RealSense camera test.
Works on Windows, macOS, and Linux (incl. WSL 2 USB-IP passthrough).

Requirements:
  pip install pyrealsense2 opencv-python
"""

import sys
import cv2
import numpy as np
import pyrealsense2 as rs


def find_device_or_exit():
    ctx = rs.context()
    if len(ctx.devices) == 0:
        sys.exit("❌  No RealSense devices found.")
    dev = ctx.devices[0]
    sn  = dev.get_info(rs.camera_info.serial_number)
    print(f"✅  Found RealSense device: {sn}")
    return sn


def main():
    find_device_or_exit()

    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        sys.exit(f"❌  Failed to start pipeline: {e}")

    align = rs.align(rs.stream.color)   # simple depth-to-RGB alignment

    print("▶  Streaming…  Press q to quit.")
    try:
        while True:
            frames          = pipeline.wait_for_frames()
            aligned_frames  = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Colourise the depth map for easy viewing
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET,
            )

            stacked = np.hstack((color_image, depth_colormap))
            cv2.imshow("RealSense RGB  |  Depth", stacked)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("⏹  Stopped.")


if __name__ == "__main__":
    main()
