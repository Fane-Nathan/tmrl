"""Offline regression tests; these never connect to or control Trackmania."""
import socket
import struct
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock

import numpy as np

from tmrl.custom.tm.utils.tools import TM2020OpenPlanetClient
from tmrl.custom.tm.utils.trajectory_assist import DemonstrationTrajectoryAssist
from tmrl.custom.tm.tm_gym_interfaces import TM2020Interface


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(1)
        self.listener.settimeout(2)
        self.client = TM2020OpenPlanetClient(port=self.listener.getsockname()[1])
        self.peer, _ = self.listener.accept()

    def tearDown(self):
        self.client.close()
        self.peer.close()
        self.listener.close()

    @staticmethod
    def packet(value):
        return struct.pack("<11f", *([value] * 11))

    def wait_for_packet(self, packet):
        condition = self.client._TM2020OpenPlanetClient__condition
        with condition:
            self.assertTrue(condition.wait_for(
                lambda: self.client._TM2020OpenPlanetClient__data == packet, timeout=1))

    def test_partial_packet_and_latest_packet_in_backlog(self):
        first, last = self.packet(1), self.packet(3)
        self.peer.sendall(first[:7])
        with self.assertRaises(TimeoutError):
            self.client.retrieve_data(timeout=0.02)
        self.peer.sendall(first[7:] + self.packet(2) + last)
        self.wait_for_packet(last)
        self.assertEqual(self.client.retrieve_data(timeout=0.1), (3.0,) * 11)

    def test_disconnect_raises_instead_of_spinning(self):
        self.peer.shutdown(socket.SHUT_RDWR)
        self.peer.close()
        with self.assertRaises(ConnectionError):
            self.client.retrieve_data(timeout=1)

    def test_missing_data_has_bounded_timeout(self):
        with self.assertRaises(TimeoutError):
            self.client.retrieve_data(timeout=0.02)

    def test_stale_packet_is_not_returned(self):
        packet = self.packet(4)
        self.peer.sendall(packet)
        self.wait_for_packet(packet)
        with self.client._TM2020OpenPlanetClient__condition:
            self.client._TM2020OpenPlanetClient__received_at -= 1.0
        with self.assertRaises(TimeoutError):
            self.client.retrieve_data(timeout=0.02, max_age=0.5)

    def test_close_is_idempotent_and_refuses_retrieval(self):
        self.client.close()
        self.client.close()
        with self.assertRaises(ConnectionError):
            self.client.retrieve_data(timeout=0.1)


class DrivingSafetyTests(unittest.TestCase):
    def test_capture_failure_releases_controls(self):
        interface = TM2020Interface.__new__(TM2020Interface)
        interface.window_interface = Mock()
        interface.window_interface.screenshot.side_effect = RuntimeError("capture failed")
        interface.send_control = Mock()
        interface._trajectory_assist_active = True
        with self.assertRaisesRegex(RuntimeError, "capture failed"):
            interface.grab_data_and_img()
        self.assertFalse(interface._trajectory_assist_active)
        np.testing.assert_array_equal(interface.send_control.call_args.args[0], np.zeros(3))

    def test_telemetry_failure_releases_controls(self):
        interface = TM2020Interface.__new__(TM2020Interface)
        interface.window_interface = Mock()
        interface.window_interface.screenshot.return_value = np.zeros((4, 4, 4), dtype=np.uint8)
        interface.client = Mock()
        interface.client.retrieve_data.side_effect = TimeoutError("telemetry stopped")
        interface.send_control = Mock()
        interface._trajectory_assist_active = True
        with self.assertRaises(TimeoutError):
            interface.grab_data_and_img()
        interface.client.retrieve_data.assert_called_once_with(timeout=0.5)
        np.testing.assert_array_equal(interface.send_control.call_args.args[0], np.zeros(3))


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "trajectory.npz"
        self.positions = np.stack((np.zeros(40), np.zeros(40), np.arange(40)), axis=1)
        self.actions = np.tile([1.0, -1.0, 0.0], (40, 1))
        self.config = {"ENABLED": True, "TRAJECTORY_PATH": str(self.path),
                       "STEERING_MODE": "pursuit", "SEARCH_FORWARD_STEPS": 32}

    def tearDown(self):
        self.directory.cleanup()

    def make_controller(self, legacy=False):
        np.savez(self.path, positions=self.positions, actions=self.actions,
                 **{"speeds_kmh" if legacy else "speeds_mps": np.full(40, 20.0)})
        return DemonstrationTrajectoryAssist(self.config)

    def test_legacy_mislabeled_speed_values_are_not_rescaled(self):
        np.testing.assert_array_equal(self.make_controller(legacy=True).speeds, np.full(40, 20.0))

    def test_straight_path_accelerates_without_steering(self):
        action, info = self.make_controller().action(np.zeros(11), np.zeros(3))
        np.testing.assert_array_equal(action, [1.0, -1.0, 0.0])
        self.assertFalse(info["emergency_stop"])

    def test_large_deviation_stops_acceleration(self):
        telemetry = np.zeros(11)
        telemetry[2] = 30.0
        action, info = self.make_controller().action(telemetry, np.ones(3))
        np.testing.assert_array_equal(action, [-1.0, 1.0, 0.0])
        self.assertTrue(info["emergency_stop"])

    def test_progress_never_rewinds(self):
        controller = self.make_controller()
        telemetry = np.zeros(11)
        telemetry[4] = 10.0
        controller.action(telemetry, np.zeros(3))
        telemetry[4] = 8.0
        _, info = controller.action(telemetry, np.zeros(3))
        self.assertEqual(info["reference_index"], 10)

    def test_nonfinite_reference_is_rejected(self):
        self.actions[3, 2] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.make_controller()

    def test_invalid_heading_filter_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "HEADING_MEASUREMENT_WEIGHT"):
            DemonstrationTrajectoryAssist({"HEADING_MEASUREMENT_WEIGHT": 0.0})


if __name__ == "__main__":
    unittest.main()
