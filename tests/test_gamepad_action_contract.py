"""No hardware is created: verify signed BC controls and unchanged legacy transport."""
import unittest
from unittest.mock import Mock

import numpy as np

from tmrl.custom.tm.utils.control_gamepad import control_gamepad, signed_trigger_controls
from tmrl.custom.tm.tm_gym_interfaces import TM2020Interface


class GamepadContractTests(unittest.TestCase):
    def test_signed_trigger_endpoints_and_midpoint(self):
        for signed, pressure in ((-1, 0), (-0.5, 0.25), (0, 0.5), (1, 1)):
            self.assertEqual(signed_trigger_controls([signed, signed, -0.3]), [pressure, pressure, -0.3])

    def test_invalid_signed_controls_are_rejected(self):
        for control in ([0, 0], [0, 0, 0, 0], [np.nan, 0, 0], [0, 2, 0], [0, 0, np.inf]):
            with self.subTest(control=control), self.assertRaises(ValueError):
                signed_trigger_controls(control)

    def test_analog_pressure_does_not_press_digital_accelerator(self):
        gamepad = Mock()
        control_gamepad(gamepad, signed_trigger_controls([0, -1, 0.2]), digital_acceleration=False)
        gamepad.right_trigger_float.assert_called_once_with(value_float=0.5)
        gamepad.left_trigger_float.assert_called_once_with(value_float=0.0)
        gamepad.press_button.assert_not_called()
        gamepad.release_button.assert_called_once_with(button=0x1000)
        gamepad.left_joystick_float.assert_called_once_with(0.2, 0.0)

    def test_legacy_defaults_are_unchanged(self):
        gamepad = Mock()
        control_gamepad(gamepad, [0.2, -1, 0.1])
        gamepad.right_trigger_float.assert_called_once_with(value_float=0.2)
        gamepad.press_button.assert_called_once_with(button=0x1000)
        gamepad = Mock()
        control_gamepad(gamepad, [-0.2, -1, 0.1])
        gamepad.right_trigger_float.assert_called_once_with(value_float=0.0)
        gamepad.press_button.assert_not_called()

    def test_signed_interface_neutral_releases_both_triggers(self):
        interface = TM2020Interface.__new__(TM2020Interface)
        interface.gamepad = interface.signed_analog_triggers = True
        interface.j = Mock()
        np.testing.assert_array_equal(interface.get_default_action(), [-1, -1, 0])
        interface.send_control(interface.get_default_action())
        interface.j.right_trigger_float.assert_called_once_with(value_float=0.0)
        interface.j.left_trigger_float.assert_called_once_with(value_float=0.0)
        interface.j.press_button.assert_not_called()

    def test_capture_failure_releases_signed_interface(self):
        interface = TM2020Interface.__new__(TM2020Interface)
        interface.gamepad = interface.signed_analog_triggers = True
        interface.j, interface.window_interface = Mock(), Mock()
        interface.window_interface.screenshot.side_effect = RuntimeError("lost camera")
        with self.assertRaises(RuntimeError):
            interface.grab_data_and_img()
        interface.j.right_trigger_float.assert_called_once_with(value_float=0.0)
        interface.j.left_trigger_float.assert_called_once_with(value_float=0.0)


if __name__ == "__main__":
    unittest.main()
