# standard library imports
import platform
import time

if platform.system() == "Windows":

    try:
        from pyautogui import click, mouseUp
    except ImportError:
        click = None
        mouseUp = None

    def _require_pyautogui():
        global click, mouseUp
        if click is None or mouseUp is None:
            from pyautogui import click as _click, mouseUp as _mouse_up
            click = _click
            mouseUp = _mouse_up

    def mouse_close_finish_pop_up_tm20(small_window=False):
        _require_pyautogui()
        if small_window:
            click(138, 108)
        else:
            click(550, 320)  # clicks where the "improve" button is supposed to be
        mouseUp()

    def mouse_save_replay_tm20(small_window=False):
        _require_pyautogui()
        time.sleep(5.0)
        if small_window:
            click(130, 110)
            mouseUp()
            time.sleep(0.2)
            click(130, 108)
            mouseUp()
        else:
            click(500, 345)
            mouseUp()
            time.sleep(0.2)
            click(500, 320)
            mouseUp()

else:

    def mouse_close_finish_pop_up_tm20(small_window=False):
        pass

    def mouse_save_replay_tm20(small_window=False):
        pass


if __name__ == "__main__":
    mouse_save_replay_tm20()
