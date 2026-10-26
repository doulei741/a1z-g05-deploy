"""Camera failure recovery must remain bounded and release rejected handles."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'a1z_g05_deploy/client'))
import dual_arm_client as client


def capture(opened=True, shape=(480, 640, 3)):
    return SimpleNamespace(isOpened=lambda: opened, set=Mock(),
                           read=lambda: (True, np.zeros(shape, np.uint8)), release=Mock())


def setup_cv(monkeypatch, captures):
    cv = SimpleNamespace(VideoCapture=Mock(side_effect=captures), CAP_V4L2=200,
                         CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4,
                         CAP_PROP_FPS=5, CAP_PROP_BUFFERSIZE=38)
    monkeypatch.setitem(sys.modules, 'cv2', cv)
    monkeypatch.setattr(client.time, 'sleep', Mock())
    return cv


def test_camera_open_retries_failed_handle_then_returns_live_capture(monkeypatch):
    failed, good = capture(False), capture()
    cv = setup_cv(monkeypatch, [failed, good])
    assert client.open_rgb_camera(12, [640, 480], 30) is good
    assert cv.VideoCapture.call_count == 2
    cv.VideoCapture.assert_called_with(12, cv.CAP_V4L2)
    failed.release.assert_called_once()
    good.release.assert_not_called()


@pytest.mark.parametrize('opened,shape', [(False, (480, 640, 3)), (True, (480, 640))])
def test_camera_open_stops_after_three_failures_and_releases_all(monkeypatch, opened, shape):
    rejected = [capture(opened, shape) for _ in range(3)]
    cv = setup_cv(monkeypatch, rejected)
    with pytest.raises(RuntimeError, match='after 3 attempts'):
        client.open_rgb_camera(12, [640, 480], 30)
    assert cv.VideoCapture.call_count == 3
    for cap in rejected:
        cap.release.assert_called_once()
