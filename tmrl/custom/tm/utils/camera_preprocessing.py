"""Shared camera conversion without copying a full-resolution BGRA slice."""
import cv2
import numpy as np


def preprocess_camera_frame(frame, resize_to=None, grayscale=True):
    """Resize contiguous BGR/BGRA first; discard alpha at policy resolution.

    Slicing a large BGRA capture to ``frame[:, :, :3]`` before cv2.resize creates
    a non-contiguous view. OpenCV then copies every source pixel just to resize
    to a tiny policy image. Keeping four channels avoids that hidden copy and
    preserves the original BGR interpolation/grayscale pixel values.
    """
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] not in (3, 4) or not frame.size:
        raise ValueError("Expected a nonempty BGR or BGRA camera frame")
    if resize_to is not None:
        frame = cv2.resize(frame, resize_to)
    if grayscale:
        code = cv2.COLOR_BGRA2GRAY if frame.shape[2] == 4 else cv2.COLOR_BGR2GRAY
    else:
        code = cv2.COLOR_BGRA2RGB if frame.shape[2] == 4 else cv2.COLOR_BGR2RGB
    return cv2.cvtColor(frame, code)
