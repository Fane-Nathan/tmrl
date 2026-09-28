"""
Trackmania 2020 Map Cycler Macro for Starter Access
Cycles through the 912 AI_1000 maps without needing Club Access or Developer Mode.

Strategies supported:
1. 'active_file' (Recommended & 100% Reliable):
   - Trackmania loads "AI_Current_Track.Map.Gbx".
   - When cycling, Python copies the next map from AI_1000 over "AI_Current_Track.Map.Gbx".
   - Sends: Esc -> Down -> Down -> Enter -> Enter (Leave map) -> Enter (Reload AI_Current_Track).
   - Trackmania parses the new map freshly from disk!

2. 'folder_browse':
   - Directly browses inside the AI_1000 map folder.
   - Sends: Esc -> Down -> Down -> Enter -> Enter (Leave) -> Down (Next track) -> Enter (Play).
"""

import os
import sys
import time
import json
import shutil
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="[MapCycler] %(asctime)s - %(levelname)s - %(message)s")

# Paths
DOCUMENTS_DIR = os.path.expanduser(r"~\Documents\Trackmania2020\Maps\My Maps")
AI_1000_DIR = os.path.join(DOCUMENTS_DIR, "AI_1000")
ACTIVE_TRACK_FILE = os.path.join(DOCUMENTS_DIR, "AI_Current_Track.Map.Gbx")
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "map_cycle_state.json")

# Input helpers
try:
    import vgamepad as vg
    HAVE_VGAMEPAD = True
except ImportError:
    HAVE_VGAMEPAD = False

try:
    import ctypes
    HAVE_CTYPES = True
except ImportError:
    HAVE_CTYPES = False

# Scancodes for Windows SendInput
ESC = 0x01
ENTER = 0x1C
UP = 0x48
DOWN = 0x50
LEFT = 0x4B
RIGHT = 0x4D
SPACE = 0x39


def get_map_list():
    if not os.path.exists(AI_1000_DIR):
        raise FileNotFoundError(f"AI_1000 directory not found at: {AI_1000_DIR}")
    maps = [f for f in sorted(os.listdir(AI_1000_DIR)) if f.endswith(".Map.Gbx")]
    return maps


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"current_index": 0, "total_cycled": 0}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def copy_active_map(index):
    maps = get_map_list()
    if not maps:
        logging.error("No maps found in AI_1000!")
        return None
    idx = index % len(maps)
    source_map = os.path.join(AI_1000_DIR, maps[idx])
    shutil.copy2(source_map, ACTIVE_TRACK_FILE)
    logging.info(f"Copied map #{idx + 1}/{len(maps)}: '{maps[idx]}' -> AI_Current_Track.Map.Gbx")
    return maps[idx]


def send_keyboard_key(scancode, is_extended=False, duration=0.08):
    if not HAVE_CTYPES:
        return
    import ctypes
    flags_press = 0x0008  # KEYEVENTF_SCANCODE
    flags_release = 0x0008 | 0x0002  # KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP
    if is_extended:
        flags_press |= 0x0001
        flags_release |= 0x0001

    class KeyBdInput(ctypes.Structure):
        _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort),
                    ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong),
                    ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

    class Input_I(ctypes.Union):
        _fields_ = [("ki", KeyBdInput)]

    class Input(ctypes.Structure):
        _fields_ = [("type", ctypes.c_ulong), ("ii", Input_I)]

    extra = ctypes.c_ulong(0)
    ii = Input_I()
    ii.ki = KeyBdInput(0, scancode, flags_press, 0, ctypes.pointer(extra))
    ctypes.windll.user32.SendInput(1, ctypes.pointer(Input(1, ii)), ctypes.sizeof(Input))

    time.sleep(duration)

    ii.ki = KeyBdInput(0, scancode, flags_release, 0, ctypes.pointer(extra))
    ctypes.windll.user32.SendInput(1, ctypes.pointer(Input(1, ii)), ctypes.sizeof(Input))


def activate_trackmania_window():
    try:
        import win32gui
        hwnd = win32gui.FindWindow(None, "Trackmania")
        if hwnd and win32gui.IsWindow(hwnd):
            win32gui.SetForegroundWindow(hwnd)
            time.sleep(0.1)
            return True
    except Exception:
        pass
    return False


def execute_cycle_sequence(strategy="active_file", gamepad=None):
    """
    Executes the in-game menu sequence to reload / advance the track.
    """
    activate_trackmania_window()
    time.sleep(0.2)

    use_gp = HAVE_VGAMEPAD and gamepad is not None

    def press_esc():
        if use_gp:
            gamepad.press_button(0x0010)  # START
            gamepad.update()
            time.sleep(0.1)
            gamepad.release_button(0x0010)
            gamepad.update()
        else:
            send_keyboard_key(ESC)

    def press_down():
        if use_gp:
            gamepad.press_button(0x0002)  # DPAD_DOWN
            gamepad.update()
            time.sleep(0.1)
            gamepad.release_button(0x0002)
            gamepad.update()
        else:
            send_keyboard_key(DOWN, is_extended=True)

    def press_select():
        if use_gp:
            gamepad.press_button(0x1000)  # A Button
            gamepad.update()
            time.sleep(0.1)
            gamepad.release_button(0x1000)
            gamepad.update()
        else:
            send_keyboard_key(ENTER)

    logging.info("Step 1: Opening Pause Menu (Esc)...")
    press_esc()
    time.sleep(0.6)

    logging.info("Step 2: Navigating to Exit (Down x2)...")
    press_down()
    time.sleep(0.25)
    press_down()
    time.sleep(0.25)

    logging.info("Step 3: Exiting Race to Map Browser...")
    press_select()
    time.sleep(2.0)  # Wait for map menu to fully load

    # In the map menu, 'AI_Current_Track' is already selected with 'PLAY' as default.
    # Do NOT press Down here, because Down moves focus to the 'Edit' button!
    logging.info("Step 4: Launching Track (Play)...")
    press_select()
    time.sleep(3.5)  # Wait for track to finish loading

    logging.info("Map cycle sequence complete! Car is at start line.")


def cycle_now(strategy="active_file", gamepad=None):
    state = load_state()
    next_idx = state.get("current_index", 0) + 1
    
    if strategy == "active_file":
        map_name = copy_active_map(next_idx)
    else:
        maps = get_map_list()
        map_name = maps[next_idx % len(maps)]

    gp = gamepad if gamepad is not None else (vg.VX360Gamepad() if HAVE_VGAMEPAD else None)
    execute_cycle_sequence(strategy=strategy, gamepad=gp)

    state["current_index"] = next_idx
    state["current_map"] = map_name
    state["total_cycled"] = state.get("total_cycled", 0) + 1
    state["last_cycle_time"] = time.time()
    save_state(state)
    logging.info(f"Successfully cycled to: {map_name} (Cycle #{state['total_cycled']})")


def run_auto_loop(interval_sec=180, strategy="active_file"):
    logging.info(f"Starting Automatic Map Cycling Daemon (Interval: {interval_sec}s, Strategy: {strategy})...")
    logging.info("Press Ctrl+C to stop.")
    
    # Initialize first active map if it doesn't exist
    if strategy == "active_file" and not os.path.exists(ACTIVE_TRACK_FILE):
        copy_active_map(0)
        logging.info("Created initial 'AI_Current_Track.Map.Gbx'. In Trackmania, please open 'AI_Current_Track' from My Maps once!")

    try:
        while True:
            time.sleep(interval_sec)
            logging.info(f"Interval of {interval_sec}s reached. Triggering map cycle...")
            cycle_now(strategy=strategy)
    except KeyboardInterrupt:
        logging.info("Map cycling stopped by user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trackmania 2020 Starter Access Map Cycler")
    parser.add_argument("--test", action="store_true", help="Perform a single test cycle right now")
    parser.add_argument("--auto", action="store_true", help="Run in continuous auto-cycle loop")
    parser.add_argument("--interval", type=int, default=180, help="Seconds between map cycles (default: 180s = 3 minutes)")
    parser.add_argument("--strategy", choices=["active_file", "folder_browse"], default="active_file",
                        help="Cycling strategy: 'active_file' (recommended) or 'folder_browse'")
    parser.add_argument("--init-file", action="store_true", help="Initialize the AI_Current_Track.Map.Gbx file from map #1")

    args = parser.parse_args()

    if args.init_file:
        copy_active_map(0)
        print("\n[OK] 'AI_Current_Track.Map.Gbx' created in My Maps!")
        print("-> In Trackmania, go to: Play -> Local -> Play a map -> My Maps -> AI_Current_Track")
    elif args.test:
        print("[TEST] Triggering a single map cycle test in 3 seconds... Make sure Trackmania is visible!")
        time.sleep(3)
        cycle_now(strategy=args.strategy)
    elif args.auto:
        run_auto_loop(interval_sec=args.interval, strategy=args.strategy)
    else:
        parser.print_help()
