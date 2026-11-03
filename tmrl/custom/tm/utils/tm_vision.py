#!/usr/bin/env python3
"""
High-Performance Real-Time Game Window Vision & Visual Tokenizer for TrackMania 2020.
Captures the TrackMania game viewport and extracts 128-dimensional visual tokens
using a lightweight convolutional neural network.
"""

import math
import platform
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if platform.system() == "Windows":
    import win32gui
    import win32ui
    import win32con


class FastWindowCapture:
    """
    Ultra-low-latency TrackMania viewport grabber using Windows GDI BitBlt.
    Captures, resizes, and converts game frames to normalized tensors in <1.5 ms.
    """
    def __init__(self, window_name: str = "Trackmania", target_size: Tuple[int, int] = (96, 96)):
        self.window_name = window_name
        self.target_w, self.target_h = target_size
        self.hwnd = None
        self.w_diff = 16
        self.h_diff = 39
        self.borders = (8, 31)
        self._find_window()

    def _find_window(self):
        if platform.system() != "Windows":
            return
        self.hwnd = win32gui.FindWindow(None, self.window_name)
        if self.hwnd == 0:
            try:
                import ctypes
                u = ctypes.windll.user32
                h_desk = u.OpenInputDesktop(0, False, 0x01FF)
                if h_desk:
                    u.SetThreadDesktop(h_desk)
                WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
                found_h = []
                def enum_cb(h, _):
                    buf = ctypes.create_unicode_buffer(512)
                    u.GetWindowTextW(h, buf, 512)
                    if buf.value and self.window_name.lower() in buf.value.lower():
                        found_h.append(h)
                    return True
                u.EnumDesktopWindows(h_desk, WNDENUMPROC(enum_cb), 0)
                if found_h:
                    self.hwnd = int(found_h[0])
            except Exception:
                pass

        if self.hwnd == 0:
            def enum_cb(hwnd, _):
                if win32gui.IsWindowVisible(hwnd):
                    text = win32gui.GetWindowText(hwnd)
                    if "trackmania" in text.lower():
                        self.hwnd = hwnd
            try:
                win32gui.EnumWindows(enum_cb, None)
            except Exception:
                pass

        if self.hwnd != 0:
            try:
                wr = win32gui.GetWindowRect(self.hwnd)
                cr = win32gui.GetClientRect(self.hwnd)
                if cr[2] > 0 and cr[3] > 0:
                    self.w_diff = wr[2] - wr[0] - cr[2] + cr[0]
                    self.h_diff = wr[3] - wr[1] - cr[3] + cr[1]
                    self.borders = (self.w_diff // 2, self.h_diff - self.w_diff // 2)
            except Exception:
                pass

    def move_and_resize(self, x: int = 1, y: int = 0, w: int = 512, h: int = 256):
        """
        Snaps TrackMania window to a compact rectangle (default 512x256 at top-left).
        Allows full desktop multitasking and terminal viewing without window minimizing.
        """
        if self.hwnd is None or self.hwnd == 0:
            self._find_window()
        if self.hwnd != 0:
            try:
                import ctypes
                ctypes.windll.user32.MoveWindow(self.hwnd, x, y, w + self.w_diff, h + self.h_diff, True)
                print(f"[+] Snapped TrackMania window to {w}x{h} at ({x}, {y})")
            except Exception:
                try:
                    win32gui.MoveWindow(self.hwnd, x, y, w + self.w_diff, h + self.h_diff, True)
                    print(f"[+] Snapped TrackMania window to {w}x{h} at ({x}, {y})")
                except Exception:
                    pass

    def grab_frame(self, grayscale: bool = True) -> np.ndarray:
        """
        Grabs the current game frame and returns a (H, W) or (H, W, 3) uint8 numpy array.
        Returns a black frame if the window cannot be captured.
        """
        if self.hwnd is None or self.hwnd == 0:
            self._find_window()
            if self.hwnd == 0:
                return np.zeros((self.target_h, self.target_w), dtype=np.uint8)

        try:
            x, y, x1, y1 = win32gui.GetWindowRect(self.hwnd)
            w = max(1, x1 - x - self.w_diff)
            h = max(1, y1 - y - self.h_diff)

            hdc = win32gui.GetWindowDC(self.hwnd)
            dc = win32ui.CreateDCFromHandle(hdc)
            memdc = dc.CreateCompatibleDC()
            bitmap = win32ui.CreateBitmap()
            bitmap.CreateCompatibleBitmap(dc, w, h)
            oldbmp = memdc.SelectObject(bitmap)
            memdc.BitBlt((0, 0), (w, h), dc, self.borders, win32con.SRCCOPY)

            bits = bitmap.GetBitmapBits(True)
            img = np.frombuffer(bits, dtype=np.uint8)
            img.shape = (h, w, 4)

            # Cleanup GDI handles to prevent leaks
            memdc.SelectObject(oldbmp)
            win32gui.DeleteObject(bitmap.GetHandle())
            memdc.DeleteDC()
            dc.DeleteDC()
            win32gui.ReleaseDC(self.hwnd, hdc)

            # Resize to target resolution (96x96)
            resized = cv2.resize(img[:, :, :3], (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)

            if grayscale:
                return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            else:
                return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        except Exception:
            return np.zeros((self.target_h, self.target_w), dtype=np.uint8)

    def grab_tensor(self, device: str = "cuda") -> torch.Tensor:
        """Returns normalized float tensor (1, 1, H, W) in [0.0, 1.0]."""
        frame = self.grab_frame(grayscale=True)
        t = torch.from_numpy(frame).float().unsqueeze(0).unsqueeze(0) / 255.0
        return t.to(device)


class VisualEncoder(nn.Module):
    """
    Lightweight 4-layer Convolutional Tokenizer for 96x96 road images.
    Converts (B, 1, 96, 96) grayscale frames into (B, embed_dim) visual tokens.
    Operates in <0.5 ms on Tensor Cores.
    """
    def __init__(self, in_channels: int = 1, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim

        # 96x96 -> 23x23 -> 10x10 -> 8x8
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # 64 * 8 * 8 = 4096 -> embed_dim
        self.fc = nn.Linear(64 * 8 * 8, embed_dim)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) or (B, T, C, H, W)
        Returns: (B, embed_dim) or (B, T, embed_dim)
        """
        orig_shape = x.shape
        if x.dim() == 5:
            B, T, C, H, W = orig_shape
            x = x.view(B * T, C, H, W)
        else:
            B, T = orig_shape[0], 1

        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = h.reshape(h.size(0), -1)
        out = self.ln(self.fc(h))

        if len(orig_shape) == 5:
            out = out.view(B, T, self.embed_dim)
        return out
