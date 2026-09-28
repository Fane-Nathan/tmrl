# standard library imports
import platform
import math


def signed_trigger_controls(control):
    """Convert BC's signed triggers to physical [0, 1] pressure, retaining steer."""
    if len(control) != 3 or not all(math.isfinite(float(c)) and -1 <= c <= 1 for c in control):
        raise ValueError("Expected three finite signed controls in [-1, 1]")
    return [(float(control[0]) + 1) / 2, (float(control[1]) + 1) / 2, float(control[2])]

if platform.system() in ("Windows", "Linux"):

    import time

    def control_gamepad(gamepad, control, digital_acceleration=True):
        assert all(-1.0 <= c <= 1.0 for c in control), "This function accepts only controls between -1.0 and 1.0"
        if control[0] > 0:  # gas
            gamepad.right_trigger_float(value_float=control[0])
            if digital_acceleration:
                gamepad.press_button(button=0x1000)  # Button A, legacy keyboard-compatible mode
            else:
                gamepad.release_button(button=0x1000)  # Never override analog pressure with full gas
        else:
            gamepad.right_trigger_float(value_float=0.0)
            gamepad.release_button(button=0x1000)
        if control[1] > 0:  # break
            gamepad.left_trigger_float(value_float=control[1])
        else:
            gamepad.left_trigger_float(value_float=0.0)
        gamepad.left_joystick_float(control[2], 0.0)  # turn
        gamepad.update()

    def gamepad_reset(gamepad):
        gamepad.reset()
        gamepad.press_button(button=0x2000)  # press B button
        gamepad.update()
        time.sleep(0.1)
        gamepad.release_button(button=0x2000)  # release B button
        gamepad.update()

    def gamepad_save_replay_tm20(gamepad):
        time.sleep(5.0)
        gamepad.reset()
        gamepad.press_button(0x0002)  # dpad down
        gamepad.update()
        time.sleep(0.1)
        gamepad.release_button(0x0002)  # dpad down
        gamepad.update()
        time.sleep(0.2)
        gamepad.press_button(0x1000)  # A
        gamepad.update()
        time.sleep(0.1)
        gamepad.release_button(0x1000)  # A
        gamepad.update()
        time.sleep(0.2)
        gamepad.press_button(0x0001)  # dpad up
        gamepad.update()
        time.sleep(0.1)
        gamepad.release_button(0x0001)  # dpad up
        gamepad.update()
        time.sleep(0.2)
        gamepad.press_button(0x1000)  # A
        gamepad.update()
        time.sleep(0.1)
        gamepad.release_button(0x1000)  # A
        gamepad.update()

    def gamepad_close_finish_pop_up_tm20(gamepad):
        gamepad.reset()
        gamepad.press_button(0x1000)  # A
        gamepad.update()
        time.sleep(0.1)
        gamepad.release_button(0x1000)  # A
        gamepad.update()

else:

    def control_gamepad(gamepad, control, digital_acceleration=True):
        pass

    def gamepad_reset(gamepad):
        pass

    def gamepad_save_replay_tm20(gamepad):
        pass

    def gamepad_close_finish_pop_up_tm20(gamepad):
        pass
