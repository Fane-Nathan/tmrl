import time
import numpy as np
from tmrl.custom.tm.tm_gym_interfaces import TM2020Interface
import tmrl.config.config_constants as cfg

env = TM2020Interface(
    img_hist_len=cfg.IMG_HIST_LEN,
    gamepad=True,
    grayscale=cfg.GRAYSCALE,
    resize_to=(cfg.IMG_WIDTH, cfg.IMG_HEIGHT),
)

env.initialize()
env.reset()
time.sleep(0.5)

positions = []
print("Accelerating straight forward (gas=1.0, brake=-1.0, steer=0.0)...")

for step in range(100):
    data, img = env.grab_data_and_img()
    pos = data[2:5]
    spd = data[0]
    positions.append(pos)
    if step % 20 == 0:
        print(f"Step {step:3d} | Speed: {spd:6.1f} km/h | Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
    env.send_control(np.array([1.0, -1.0, 0.0], dtype=np.float32))
    time.sleep(0.05)

env.send_control(np.array([-1.0, -1.0, 0.0], dtype=np.float32))
env.reset()
env.close()

p0 = positions[0]
p_end = positions[-1]
disp = np.linalg.norm(np.array(p_end) - np.array(p0))
print(f"Total straight displacement: {disp:.2f} m")
print(f"Start: {p0} -> End: {p_end}")
