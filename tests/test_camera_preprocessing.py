"""Pixel parity between legacy and copy-free camera preprocessing."""
import unittest
from unittest.mock import Mock, patch
import platform
import sys
from types import SimpleNamespace

import cv2
import numpy as np

from tmrl.custom.tm.utils.camera_preprocessing import preprocess_camera_frame


class CameraPreprocessingTests(unittest.TestCase):
    @unittest.skipUnless(platform.system() == "Windows", "Windows DXcam initialization contract")
    def test_preserved_capture_never_calls_window_resizer_on_reinitialization(self):
        from tmrl.custom.tm.utils import window
        camera = Mock()
        camera.get_latest_frame.return_value = np.zeros((2, 2, 4), dtype=np.uint8)
        fake_dxcam = SimpleNamespace(create=Mock(return_value=camera))
        region = (0, 0, 2560, 1600)
        with patch.multiple(window, _DXCAM_CAMERA=None, _DXCAM_REGION=None, _DXCAM_STARTED=False), \
                patch.dict(sys.modules, {"dxcam": fake_dxcam}), \
                patch.object(window.cfg, "CAPTURE_BACKEND", "dxcam"), \
                patch.object(window.win32gui, "FindWindow", return_value=123), \
                patch.object(window, "_physical_client_region", return_value=region), \
                patch.object(window, "_prepare_trackmania_window") as resize:
            self.assertIs(window.preinitialize_dxcam_capture(resize_window=False), camera)
            # Environment/actor construction must reuse the established stream
            # even if another caller invokes the legacy default signature.
            self.assertIs(window.preinitialize_dxcam_capture(), camera)
            resize.assert_not_called()
            self.assertEqual(camera.start.call_args.kwargs["region"], region)
            fake_dxcam.create.assert_called_once()

    def test_bgra_keeps_exact_grayscale_and_rgb_pixels(self):
        frame = np.random.default_rng(42).integers(0, 256, size=(360, 640, 4), dtype=np.uint8)
        for size in ((96, 96), (64, 48), None):
            bgr = frame[:, :, :3]
            if size is not None:
                bgr = cv2.resize(bgr, size)
            np.testing.assert_array_equal(preprocess_camera_frame(frame, size, True),
                                          cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
            np.testing.assert_array_equal(preprocess_camera_frame(frame, size, False), bgr[:, :, ::-1])

    def test_three_channel_captures_also_work(self):
        frame = np.random.default_rng(9).integers(0, 256, size=(256, 512, 3), dtype=np.uint8)
        expected = cv2.cvtColor(cv2.resize(frame, (96, 96)), cv2.COLOR_BGR2GRAY)
        np.testing.assert_array_equal(preprocess_camera_frame(frame, (96, 96)), expected)

    def test_invalid_frame_is_rejected(self):
        for shape in ((0, 0, 4), (96, 96), (96, 96, 2)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                preprocess_camera_frame(np.zeros(shape, dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
