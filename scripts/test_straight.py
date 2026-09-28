import time
from tmrl.custom.tm.utils.tools import TM2020OpenPlanetClient
from tmrl.custom.tm.utils.gamepad import Gamepad

client = TM2020OpenPlanetClient()
gp = Gamepad()
time.sleep(0.5)

d0 = client.retrieve_data()
print('Start pos:', d0[2:5], 'speed:', d0[0])

# Reset car to start line
gp.reset_forward()
time.sleep(1.0)

# Accelerate straight forward for 30 steps (1.5 seconds)
for _ in range(30):
    gp.accelerate(1.0)
    gp.steer(0.0)
    time.sleep(0.05)

d1 = client.retrieve_data()
print('After 1.5s gas straight pos:', d1[2:5], 'speed:', d1[0])

# Release controls and reset
gp.accelerate(0.0)
gp.reset_forward()
time.sleep(0.5)
gp.close()
