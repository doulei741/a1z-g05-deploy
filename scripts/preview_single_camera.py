# This file is part of G05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Preview one camera from the single-arm collection config, without starting the robot."""

import argparse
import os
from pathlib import Path

import cv2
import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "collect_single_left.local.yaml"
CAMERAS = {
    "head": ("cam_high", "Head camera"),
    "wrist": ("cam_left_wrist", "Wrist camera"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("camera", choices=CAMERAS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Local single-arm collection YAML")
    parser.add_argument("--check-only", action="store_true", help="read one frame without opening a window")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    key, title = CAMERAS[args.camera]
    path = cfg["camera_cfg"]["topics"][key]
    width = cfg["camera_cfg"]["width"]
    height = cfg["camera_cfg"]["height"]
    fps = cfg["collector_cfg"]["camera_freq"]
    if not os.path.exists(path):
        raise RuntimeError(f"Camera missing: {path}")

    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera: {path} (stop collection first)")
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        print(f"{title}: {path} -> {os.path.realpath(path)}", flush=True)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Cannot read frame from {path}")
        print(f"Frame: {frame.shape[1]}x{frame.shape[0]}", flush=True)
        if args.check_only:
            return

        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, 960, 720)
        print("Press Q or Esc in the preview window to close.", flush=True)
        while True:
            cv2.imshow(title, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Camera stream stopped: {path}")
    finally:
        cap.release()
        if not args.check_only:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
