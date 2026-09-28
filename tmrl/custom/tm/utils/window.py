import atexit
import logging

import platform

import numpy as np

import tmrl.config.config_constants as cfg


if platform.system() == "Windows":

    import ctypes

    try:
        hwinsta = ctypes.windll.user32.OpenWindowStationW("WinSta0", False, 0x10000000)
        if hwinsta:
            ctypes.windll.user32.SetProcessWindowStation(hwinsta)
        hdesk = ctypes.windll.user32.OpenDesktopW("Default", 0, False, 0x10000000)
        if hdesk:
            ctypes.windll.user32.SetThreadDesktop(hdesk)
    except Exception:
        pass

    import win32gui
    import win32ui
    import win32con

    _DXCAM_CAMERA = None
    _DXCAM_REGION = None
    _DXCAM_STARTED = False


    def _dpi_scale(hwnd):
        try:
            awareness = ctypes.c_int()
            ctypes.windll.shcore.GetProcessDpiAwareness(
                0, ctypes.byref(awareness)
            )
            if awareness.value != 0:
                # Coordinates are already physical pixels for DPI-aware callers.
                return 1.0
            return max(
                1.0,
                float(ctypes.windll.user32.GetDpiForWindow(hwnd)) / 96.0,
            )
        except (AttributeError, OSError):
            return 1.0


    def _physical_client_region(hwnd):
        cr = win32gui.GetClientRect(hwnd)
        left, top = win32gui.ClientToScreen(hwnd, (cr[0], cr[1]))
        right, bottom = win32gui.ClientToScreen(hwnd, (cr[2], cr[3]))
        scale = _dpi_scale(hwnd)
        region = tuple(
            int(round(value * scale))
            for value in (left, top, right, bottom)
        )
        if region[2] <= region[0] or region[3] <= region[1]:
            raise RuntimeError(
                f"Trackmania client rectangle is not capturable: {region}"
            )
        return region


    def _compatible_capture_region(actual, expected, tolerance=1):
        """Allow DWM's one-pixel client-size rounding, never window movement."""
        if actual is None or expected is None:
            return False
        same_origin = actual[:2] == expected[:2]
        size_within_tolerance = all(
            abs(actual[index] - expected[index]) <= tolerance
            for index in (2, 3)
        )
        return same_origin and size_within_tolerance


    def _attach_to_interactive_desktop():
        try:
            hwinsta = ctypes.windll.user32.OpenWindowStationW("WinSta0", False, 0x10000000)
            if hwinsta:
                ctypes.windll.user32.SetProcessWindowStation(hwinsta)
            hdesk = ctypes.windll.user32.OpenDesktopW("Default", 0, False, 0x10000000)
            if hdesk:
                ctypes.windll.user32.SetThreadDesktop(hdesk)
        except Exception:
            pass


    def _prepare_trackmania_window(window_name="Trackmania"):
        """Place the client at the configured capture location before streaming."""
        _attach_to_interactive_desktop()
        hwnd = win32gui.FindWindow(None, window_name)
        if not hwnd:
            raise RuntimeError(f"Could not find a window named {window_name}.")
        wr = win32gui.GetWindowRect(hwnd)
        cr = win32gui.GetClientRect(hwnd)
        w_diff = wr[2] - wr[0] - cr[2] + cr[0]
        h_diff = wr[3] - wr[1] - cr[3] + cr[1]
        x = -w_diff // 2
        y = 0
        w = int(cfg.WINDOW_WIDTH) + w_diff
        h = int(cfg.WINDOW_HEIGHT) + h_diff
        if getattr(cfg, "CAPTURE_KEEP_ON_TOP", False):
            win32gui.SetWindowPos(
                hwnd,
                win32con.HWND_TOPMOST,
                x,
                y,
                w,
                h,
                win32con.SWP_SHOWWINDOW,
            )
        else:
            win32gui.MoveWindow(hwnd, x, y, w, h, True)
        # DWM can clamp a borderless window by a pixel or two after MoveWindow.
        import time

        time.sleep(0.05)
        return hwnd, _physical_client_region(hwnd)


    def _release_dxcam():
        global _DXCAM_CAMERA, _DXCAM_REGION, _DXCAM_STARTED
        if _DXCAM_CAMERA is not None:
            try:
                if _DXCAM_STARTED:
                    _DXCAM_CAMERA.stop()
            except Exception:
                pass
            try:
                _DXCAM_CAMERA.release()
            except Exception:
                pass
            _DXCAM_CAMERA = None
            _DXCAM_REGION = None
            _DXCAM_STARTED = False


    atexit.register(_release_dxcam)


    def preinitialize_dxcam_capture(required=None, resize_window=True):
        """Create Desktop Duplication before PyTorch initializes GPU libraries.

        On some hybrid-GPU laptops, importing/initializing PyTorch first makes
        DXGI choose an incompatible adapter.  CLI entry points call this helper
        before importing the torch-dependent configuration graph.

        Set resize_window=False to capture the existing client unchanged. Moving
        an exclusive/fullscreen game to the small training rectangle can leave
        a cropped, frozen swap-chain image even when client geometry looks valid.
        """
        global _DXCAM_CAMERA, _DXCAM_REGION, _DXCAM_STARTED
        requested = str(getattr(cfg, "CAPTURE_BACKEND", "gdi")).strip().lower()
        if required is None:
            required = requested == "dxcam"
        if requested not in {"dxcam", "auto"}:
            return None
        if _DXCAM_CAMERA is not None:
            return _DXCAM_CAMERA
        try:
            import dxcam

            _DXCAM_CAMERA = dxcam.create(output_color="BGRA")
            if resize_window:
                _, _DXCAM_REGION = _prepare_trackmania_window()
            else:
                hwnd = win32gui.FindWindow(None, "Trackmania")
                if not hwnd:
                    raise RuntimeError("Could not find a Trackmania window")
                _DXCAM_REGION = _physical_client_region(hwnd)
            _DXCAM_CAMERA.start(
                region=_DXCAM_REGION,
                target_fps=max(1, int(getattr(cfg, "CAPTURE_FPS", 60))),
                video_mode=True,
            )
            _DXCAM_STARTED = True
            # Do not let the first environment reset consume an empty ring.
            import time

            deadline = time.perf_counter() + 2.0
            while _DXCAM_CAMERA.get_latest_frame(copy=False) is None:
                if time.perf_counter() >= deadline:
                    raise RuntimeError("DXcam did not publish its first frame")
                time.sleep(0.005)
            return _DXCAM_CAMERA
        except Exception as exc:
            _release_dxcam()
            if required:
                raise RuntimeError(
                    "DXcam capture was explicitly requested but could not be "
                    "initialized. It must be created before PyTorch on hybrid-"
                    "GPU Windows systems; also verify dxcam>=0.3.0 is installed."
                ) from exc
            logging.warning(
                "DXcam unavailable (%s); falling back to slower GDI capture.",
                exc,
            )
            return None


    class WindowInterface:
        def __init__(self, window_name):
            self.window_name = window_name
            self._dxcam = None
            requested_backend = str(
                getattr(cfg, "CAPTURE_BACKEND", "gdi")
            ).strip().lower()
            if requested_backend not in {"gdi", "dxcam", "auto"}:
                raise ValueError(
                    "ENV.CAPTURE_BACKEND must be one of: gdi, dxcam, auto"
                )
            self.capture_backend = requested_backend

            _attach_to_interactive_desktop()
            hwnd = win32gui.FindWindow(None, self.window_name)
            assert hwnd != 0, f"Could not find a window named {self.window_name}."
            self.hwnd = hwnd

            while True:  # in case the window is reduced
                wr = win32gui.GetWindowRect(hwnd)
                cr = win32gui.GetClientRect(hwnd)
                if cr[2] > 0 and cr[3] > 0:
                    break

            self.w_diff = wr[2] - wr[0] - cr[2] + cr[0]  # (16 on W10)
            self.h_diff = wr[3] - wr[1] - cr[3] + cr[1]  # (39 on W10)

            self.borders = (self.w_diff // 2, self.h_diff - self.w_diff // 2)

            self.x_origin_offset = - self.w_diff // 2
            self.y_origin_offset = 0

            if requested_backend in {"dxcam", "auto"}:
                self._dxcam = preinitialize_dxcam_capture(
                    required=requested_backend == "dxcam"
                )
                if self._dxcam is not None:
                    self.capture_backend = "dxcam"
                    logging.info("Window capture backend: DXcam Desktop Duplication")
                else:
                    self.capture_backend = "gdi"

        def __del__(self):
            # DXcam is process-global and released through the atexit hook.
            pass

        def _dxcam_region(self, hwnd):
            """Return the physical-pixel client rectangle expected by DXGI."""
            return _physical_client_region(hwnd)

        def _screenshot_dxcam(self, hwnd):
            region = self._dxcam_region(hwnd)
            if win32gui.IsIconic(hwnd):
                raise RuntimeError(
                    "Trackmania is minimized; DXcam cannot provide valid policy frames"
                )
            if not _compatible_capture_region(region, _DXCAM_REGION):
                raise RuntimeError(
                    "Trackmania moved or resized after DXcam started. Restart the "
                    f"worker to realign capture (expected {_DXCAM_REGION}, got {region})."
                )
            frame = self._dxcam.get_latest_frame(copy=True)
            if frame is None:
                raise RuntimeError("DXcam returned no Trackmania frame")
            return frame

        def _screenshot_gdi(self, hwnd):
            while True:  # avoids crashes when the window is reduced
                x, y, x1, y1 = win32gui.GetWindowRect(hwnd)
                w = x1 - x - self.w_diff
                h = y1 - y - self.h_diff
                if w > 0 and h > 0:
                    break
            hdc = win32gui.GetWindowDC(hwnd)
            dc = win32ui.CreateDCFromHandle(hdc)
            memdc = dc.CreateCompatibleDC()
            bitmap = win32ui.CreateBitmap()
            bitmap.CreateCompatibleBitmap(dc, w, h)
            oldbmp = memdc.SelectObject(bitmap)
            try:
                # Use PrintWindow with PW_RENDERFULLCONTENT (2) for hardware-accelerated DirectX windows
                res = ctypes.windll.user32.PrintWindow(hwnd, memdc.GetSafeHdc(), 2)
                if not res:
                    memdc.BitBlt((0, 0), (w, h), dc, self.borders, win32con.SRCCOPY)
                bits = bitmap.GetBitmapBits(True)
                img = np.frombuffer(bits, dtype="uint8")
                img.shape = (h, w, 4)
                return img
            finally:
                memdc.SelectObject(oldbmp)
                win32gui.DeleteObject(bitmap.GetHandle())
                memdc.DeleteDC()
                dc.DeleteDC()
                win32gui.ReleaseDC(hwnd, hdc)

        def screenshot(self):
            hwnd = getattr(self, "hwnd", None)
            if not hwnd or not win32gui.IsWindow(hwnd):
                hwnd = win32gui.FindWindow(None, self.window_name)
                if hwnd != 0:
                    self.hwnd = hwnd
            assert hwnd != 0, f"Could not find a window named {self.window_name}."
            if self.capture_backend == "dxcam":
                return self._screenshot_dxcam(hwnd)
            return self._screenshot_gdi(hwnd)

        def move_and_resize(self, x=1, y=0, w=cfg.WINDOW_WIDTH, h=cfg.WINDOW_HEIGHT):
            if self.capture_backend == "dxcam" and _DXCAM_REGION is not None:
                if (x, y, w, h) != (1, 0, cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT):
                    raise RuntimeError("Restart DXcam before changing its capture geometry")
                # preinitialize_dxcam_capture() already positioned the window
                # before starting Desktop Duplication. Repeating MoveWindow can
                # make DWM round the client height by one pixel and needlessly
                # invalidate an otherwise aligned capture stream.
                current = self._dxcam_region(self.hwnd)
                if _compatible_capture_region(current, _DXCAM_REGION):
                    return
            x += self.x_origin_offset
            y += self.y_origin_offset
            w += self.w_diff
            h += self.h_diff
            hwnd = getattr(self, "hwnd", None)
            if not hwnd or not win32gui.IsWindow(hwnd):
                hwnd = win32gui.FindWindow(None, self.window_name)
                if hwnd != 0:
                    self.hwnd = hwnd
            assert hwnd != 0, f"Could not find a window named {self.window_name}."
            win32gui.MoveWindow(hwnd, x, y, w, h, True)


elif platform.system() == "Linux":

    import subprocess
    import time
    import mss


    def get_window_id(name):
        try:
            result = subprocess.run(['xdotool', 'search', '--onlyvisible', '--name', '.'],
                                    capture_output=True, text=True, check=True)
            window_ids = result.stdout.strip().split('\n')
            for window_id in window_ids:
                result = subprocess.run(['xdotool', 'getwindowname', window_id],
                                        capture_output=True, text=True, check=True)
                if result.stdout.strip() == name:
                    logging.debug(f"detected window {name}, id={window_id}")
                    return window_id

            logging.error(f"failed to find window '{name}'")
            raise NoSuchWindowException(name)

        except subprocess.CalledProcessError as e:
            logging.error(f"process error searching for window '{name}")
            raise NoSuchWindowException(name)


    def get_window_geometry(name):
        """
        FIXME: xdotool doesn't agree with MSS, so we use hardcoded offsets instead for now
        """
        try:
            result = subprocess.run(['xdotool', 'search', '--name', name, 'getwindowgeometry', '--shell'],
                                    capture_output=True, text=True, check=True)
            elements = result.stdout.strip().split('\n')
            res_id = None
            res_x = None
            res_y = None
            res_w = None
            res_h = None
            for elt in elements:
                low_elt = elt.lower()
                if low_elt.startswith("window="):
                    res_id = elt[7:]
                elif low_elt.startswith("x="):
                    res_x = int(elt[2:])
                elif low_elt.startswith("y="):
                    res_y = int(elt[2:])
                elif low_elt.startswith("width="):
                    res_w = int(elt[6:])
                elif low_elt.startswith("height="):
                    res_h = int(elt[7:])

            if None in (res_id, res_x, res_y, res_w, res_h):
                raise GeometrySearchException(f"Found None in window '{name}' geometry: {(res_id, res_x, res_y, res_w, res_h)}")

            return res_id, res_x, res_y, res_w, res_h

        except subprocess.CalledProcessError as e:
            logging.error(f"process error searching for {name} window geometry")
            raise e


    class NoSuchWindowException(Exception):
        """thrown if a named window can't be found"""
        pass


    class GeometrySearchException(Exception):
        """thrown if geometry search fails"""
        pass


    class WindowInterface:
        def __init__(self, window_name):
            self.sct = mss.mss()

            self.window_name = window_name
            try:
                self.window_id = get_window_id(window_name)
            except NoSuchWindowException as e:
                logging.error(f"get_window_id failed, is xdotool correctly installed? {str(e)}")
                self.window_id = None

            self.w = None
            self.h = None
            self.x = None
            self.y = None
            self.x_offset = cfg.LINUX_X_OFFSET
            self.y_offset = cfg.LINUX_Y_OFFSET

            self.process = None

        def __del__(self):
            pass
            self.sct.close()

        def execute_command(self, c):
            if self.process is None or self.process.poll() is not None:
                self.process = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                stderr=subprocess.PIPE)
            self.process.stdin.write(c.encode())
            self.process.stdin.flush()

        def screenshot(self):
            try:
                monitor = {"top": self.x + self.x_offset, "left": self.y + self.y_offset, "width": self.w, "height": self.h}
                img = np.array(self.sct.grab(monitor))
                return img

            except subprocess.CalledProcessError as e:
                logging.error(f"failed to capture screenshot")
                raise e

        def move_and_resize(self, x=0, y=0, w=cfg.WINDOW_WIDTH, h=cfg.WINDOW_HEIGHT):
            logging.debug(f"prepare {self.window_name} to {w}x{h} @ {x}, {y}")

            try:
                # debug
                c_focus = f"xdotool windowfocus {self.window_id}\n"
                self.execute_command(c_focus)

                # move
                logging.debug(f"move window {str(self.window_name)}")
                c_move = f"xdotool windowmove {str(self.window_id)} {str(x)} {str(y)}\n"
                self.execute_command(c_move)

                # resize
                logging.debug(f"resize window {str(self.window_name)}")
                c_resize = f"xdotool windowsize {str(self.window_id)} {str(w)} {str(h)}\n"
                self.execute_command(c_resize)

                self.w = w
                self.h = h
                self.x = x
                self.y = y

                # instead of using xdotool --sync, which doesn't return
                logging.debug(f"success, let me nap 1s to make sure everything computed")
                time.sleep(1)

                # # retrieve actual position of the window and set offsets
                # geo_id, geo_x, geo_y, geo_w, geo_h = get_window_geometry(self.window_name)
                #
                # if geo_id != self.window_id:
                #     raise GeometrySearchException(f"wrong geo_id: {geo_id} != {self.window_id}")
                # if geo_w != self.w:
                #     raise GeometrySearchException(f"wrong geo_w: {geo_w} != {self.w}")
                # if geo_h != self.h:
                #     raise GeometrySearchException(f"wrong geo_h: {geo_h} != {self.h}")
                #
                # self.x_offset = geo_x - self.x
                # self.y_offset = geo_y - self.y

            except subprocess.CalledProcessError as e:
                logging.error(f"failed to resize window_id '{self.window_id}'")

            except NoSuchWindowException as e:
                logging.error(f"failed to find window: {str(e)}")

            # except GeometrySearchException as e:
            #     logging.error(f"failed to retrieve window geometry: {str(e)}")


def profile_screenshot():
    from pyinstrument import Profiler
    pro = Profiler()
    window_interface = WindowInterface("Trackmania")
    pro.start()
    for _ in range(5000):
        snap = window_interface.screenshot()
    pro.stop()
    pro.print(show_all=True)


if __name__ == "__main__":
    profile_screenshot()
