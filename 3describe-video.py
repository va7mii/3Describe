# smart_hand_cube.py – RealSense + MediaPipe demo (dynamic‑resolution fix)
"""Interactive hand‑tracking demo that draws a 3‑D cube you can poke with
hand landmarks.

Fix (2025‑06‑19)
----------------
The fallback to 640 × 480 caused an **IndexError** because several parts of
the maths were hard‑coded for 1280 × 720. All image‑size literals are now
replaced with runtime vars `img_w`, `img_h`, so the code works at any
profile the camera delivers.
"""

from __future__ import annotations

import sys, math, datetime as dt
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs

# ----------------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------------

FOV_X_DEG, FOV_Y_DEG = 87.0, 58.0  # D435 typical RGB FOV
FOV_X = FOV_X_DEG / 180 * math.pi
FOV_Y = FOV_Y_DEG / 180 * math.pi

font        = cv2.FONT_HERSHEY_SIMPLEX
font_scale  = 0.5
font_colour = (0, 50, 255)
font_thick  = 1

def clamp(val: int, hi: int) -> int:
    """Clamp *val* to **[0, hi‑1]**."""
    return max(0, min(val, hi - 1))

# ----------------------------------------------------------------------------
# RealSense camera handling
# ----------------------------------------------------------------------------

class RealSenseCamera:
    def __init__(self, desired_res: tuple[int, int], fps: int):
        self.pipeline = rs.pipeline()
        self.depth_scale = None  # will be set later
        self.profile = self._start_pipeline(desired_res, fps)
        self.align   = rs.align(rs.stream.color)

    @staticmethod
    def _first_device_sn() -> str:
        ctx = rs.context()
        if not ctx.devices:
            sys.exit("❌  No RealSense devices found.")
        dev = ctx.devices[0]
        sn  = dev.get_info(rs.camera_info.serial_number)
        print("✅  Found RealSense device:", sn)
        return sn

    def _start_pipeline(self, res: tuple[int, int], fps: int) -> rs.pipeline_profile:
        w, h = res
        cfg  = rs.config()
        cfg.enable_device(self._first_device_sn())
        cfg.enable_stream(rs.stream.depth,  w, h, rs.format.z16, fps)
        cfg.enable_stream(rs.stream.color,  w, h, rs.format.bgr8, fps)

        for attempt, fallback in enumerate([(w, h), (640, 480), None], 1):
            try:
                profile = self.pipeline.start(cfg)
                break
            except RuntimeError as e:
                if fallback is None:
                    raise RuntimeError("All profiles failed") from e
                print(f"⚠  Attempt {attempt} failed ({e}). ➜ Retrying {fallback}…")
                cfg.disable_all_streams()
                f_w, f_h = fallback
                cfg.enable_stream(rs.stream.depth, f_w, f_h, rs.format.z16, fps)
                cfg.enable_stream(rs.stream.color, f_w, f_h, rs.format.bgr8, fps)
        # depth scale
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        print("ℹ️  Using RGB+Depth:", profile.get_streams()[0].as_video_stream_profile().width(), "×",
              profile.get_streams()[0].as_video_stream_profile().height(), "@", fps)
        return profile

    def frames(self):
        return self.align.process(self.pipeline.wait_for_frames())

    def stop(self):
        self.pipeline.stop()

# ----------------------------------------------------------------------------
# 3‑D cube model
# ----------------------------------------------------------------------------
class Prop:
    """A wire‑frame cube with simple rotation & translation."""

    def __init__(self, x: float, y: float, z: float, w: float, l: float, h: float):
        self.x, self.y, self.z = x, y, z
        self.w, self.l, self.h = w, l, h
        # eight corner points (local space)
        hw, hl, hh = w / 2, l / 2, h / 2
        self.obj = [np.array([sx * hw, sy * hl, sz * hh])
                    for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)]
        self.lines = [(0,1), (0,2), (3,1), (3,2),  # bottom
                      (4,5), (4,6), (7,5), (7,6),  # top
                      (0,4), (1,5), (2,6), (3,7)]  # sides
        self.colour = (0, 255, 0)
        self.radius = max(w, l, h)

    # basic rotations
    def _rotate(self, angle, axis):
        c, s = math.cos(angle), math.sin(angle)
        R = {
            'x': np.array([[1,0,0],[0,c,-s],[0,s,c]]),
            'y': np.array([[c,0,s],[0,1,0],[-s,0,c]]),
            'z': np.array([[c,-s,0],[s,c,0],[0,0,1]]),
        }[axis]
        self.obj = [R @ p.reshape(3,1) for p in self.obj]

    def rotate_x(self, a): self._rotate(a, 'x')
    def rotate_y(self, a): self._rotate(a, 'y')
    def rotate_z(self, a): self._rotate(a, 'z')

    # translations
    def translate(self, dx=0, dy=0, dz=0):
        self.x += dx; self.y += dy; self.z = max(0.05, self.z + dz)

    # projection helpers
    def _project_point(self, pt, img_w, img_h):
        ox = pt[0] + self.x
        oy = pt[1] + self.y
        oz = pt[2] + self.z
        half_w, half_h = img_w / 2, img_h / 2
        full_x = oz * math.tan(FOV_X / 2)
        full_y = oz * math.tan(FOV_Y / 2)
        px = int(half_w + ox / full_x * half_w)
        py = int(half_h - oy / full_y * half_h)
        return px, py

    def project_2d(self, img_w: int, img_h: int):
        return [self._project_point(p, img_w, img_h) for p in self.obj]

    def projected_center(self, img_w: int, img_h: int):
        return self._project_point(np.zeros(3), img_w, img_h)

# ----------------------------------------------------------------------------
# Mediapipe hands wrapper
# ----------------------------------------------------------------------------
class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands    = self.mp_hands.Hands(model_complexity=0, max_num_hands=2)
        self.drawer   = mp.solutions.drawing_utils

    def detect(self, rgb_img):
        return self.hands.process(rgb_img)

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    STREAM_RES = (1280, 720)  # try this first, will down‑shift if USB‑2
    FPS        = 30

    cam   = RealSenseCamera(STREAM_RES, FPS)
    cube  = Prop(0, 0, 0.5, 0.2, 0.2, 0.2)
    hands = HandTracker()

    print("▶  Streaming. Press q to quit …")

    try:
        while True:
            frames = cam.frames()
            depth  = np.asanyarray(frames.get_depth_frame().get_data())
            colour = np.asanyarray(frames.get_color_frame().get_data())

            # mirror for user‑friendly view
            depth_flipped  = cv2.flip(depth, 1)
            colour_flipped = cv2.flip(colour, 1)
            img_h, img_w   = depth_flipped.shape
            rgb_for_mp     = cv2.cvtColor(colour_flipped, cv2.COLOR_BGR2RGB)

            # draw cube
            img = colour_flipped.copy()
            projections = cube.project_2d(img_w, img_h)
            for a, b in cube.lines:
                ax, ay = projections[a]
                bx, by = projections[b]
                if 0 <= ay < img_h and 0 <= by < img_h:
                    if depth_flipped[ay, ax] * cam.depth_scale > cube.obj[a][2] + cube.z:
                        cv2.line(img, (ax, ay), (bx, by), cube.colour, 2)

            # hand detection
            res = hands.detect(rgb_for_mp)
            touched = False
            if res.multi_hand_landmarks:
                for h_idx, hand in enumerate(res.multi_hand_landmarks):
                    hands.drawer.draw_landmarks(img, hand, hands.mp_hands.HAND_CONNECTIONS)
                    for lm in hand.landmark:
                        x = clamp(int(lm.x * img_w), img_w)
                        y = clamp(int(lm.y * img_h), img_h)
                        z = depth_flipped[y, x] * cam.depth_scale
                        oz = cube.z
                        ox = cube.x - (z * math.tan(FOV_X/2) * ((x - img_w/2) / (img_w/2)))
                        oy = cube.y - (z * math.tan(FOV_Y/2) * ((img_h/2 - y) / (img_h/2)))
                        dist = math.sqrt(ox**2 + oy**2 + (oz - z)**2)
                        if dist <= cube.radius * 0.9:
                            touched = True
                            cube.translate(dx=0.0001*(ox/dist), dy=0.0001*(oy/dist), dz=0.0001*((oz - z)/dist))
            cube.colour = (255, 0, 0) if touched else (0, 255, 0)
            cube.rotate_x(0.01); cube.rotate_y(0.01); cube.rotate_z(0.01)

            # fps overlay
            cv2.putText(img, f"FPS: {int(cv2.getTickFrequency()/(cv2.getTickCount()))}",
                        (20, 40), font, font_scale, font_colour, font_thick)

            cv2.imshow("RealSense demo", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
