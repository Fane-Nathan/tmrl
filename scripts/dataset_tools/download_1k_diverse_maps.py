#!/usr/bin/env python3
"""
High-Speed Downloader for 1,000 Diverse TrackMania 2020 Maps from TMX.

Downloads 1,000 top-rated community maps across 10 diverse driving styles:
1. Tech (Tag 3): Tarmac, braking points, precise drifts
2. SpeedTech (Tag 7): High-speed technical flow
3. Dirt (Tag 15): Loose dirt surfaces, slide angles
4. Offroad (Tag 9): Grass, mud, dirt combinations
5. Ice (Tag 14): Low-grip ice, bobsleigh, counter-steering
6. FullSpeed (Tag 2): High speed, loops, wallrides, banked roads
7. Competitive (Tag 13): Standard esports circuits
8. Race (Tag 1): Classic racing lines
9. SpeedFun (Tag 12): Fast acrobatic transitions
10. MultiLap (Tag 8): Endurance and lap circuits

Automatically generates:
1. C:\\Users\\felix\\Documents\\Trackmania\\Maps\\AI_1000\\ (all 1,000 .Map.Gbx)
2. C:\\Users\\felix\\Documents\\Trackmania\\MatchSettings\\AI_1000_Arcade.txt (playlist)
3. D:\\Project\\tmrl\\data\\maps_1k\\manifest.json (metadata index)
"""

import concurrent.futures
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from typing import Dict, List, Set, Tuple

CATEGORIES = [
    ("Tech", 3),
    ("SpeedTech", 7),
    ("Dirt", 15),
    ("Offroad", 9),
    ("Ice", 14),
    ("FullSpeed", 2),
    ("Competitive", 13),
    ("Race", 1),
    ("SpeedFun", 12),
    ("MultiLap", 8),
]

USER_AGENT = "TMRL-DiverseMapDownloader/1.0 (Machine Learning Research)"
HEADERS = {"User-Agent": USER_AGENT}


def sanitize_filename(name: str) -> str:
    """Removes non-ascii or filesystem-illegal characters."""
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = re.sub(r"\s+", "_", name.strip())
    name = re.sub(r"[^\w\-_.]", "", name)
    return name[:40] or "track"


def fetch_category_maps(category: str, tag_id: int, seen_ids: Set[int]) -> List[Tuple[Dict, str]]:
    """Fetches up to 100 top-rated map metadata for a specific tag using TMX API."""
    url = (
        f"https://trackmania.exchange/mapsearch2/search"
        f"?api=on&format=json&tags={tag_id}&priord=2&limit=100"
    )
    results = []
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            items = data.get("results", [])
            for item in items:
                tid = item.get("TrackID")
                if tid and tid not in seen_ids and item.get("Downloadable", True):
                    seen_ids.add(tid)
                    results.append((item, category))
        print(f"[+] Cataloged {len(results):3d} maps for {category:<12} (Tag {tag_id})", flush=True)
    except Exception as e:
        print(f"[-] Error fetching category {category}: {e}", flush=True)
    return results


def download_single_map(item: Dict, category: str, output_dirs: List[str]) -> Tuple[bool, str, int]:
    """Downloads one map from TMX and saves it to all target directories."""
    track_id = item["TrackID"]
    raw_name = item.get("Name", f"map_{track_id}")
    clean_name = sanitize_filename(raw_name)
    filename = f"{category}_{track_id}_{clean_name}.Map.Gbx"

    # Check if already downloaded
    all_exist = True
    for d in output_dirs:
        p = os.path.join(d, filename)
        if not (os.path.exists(p) and os.path.getsize(p) > 500):
            all_exist = False
            break

    if all_exist:
        size = os.path.getsize(os.path.join(output_dirs[0], filename))
        return True, filename, size

    url = f"https://trackmania.exchange/maps/download/{track_id}"
    req = urllib.request.Request(url, headers=HEADERS)

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = resp.read()
                if len(data) > 200 and data.startswith(b"GBX"):
                    for d in output_dirs:
                        p = os.path.join(d, filename)
                        with open(p, "wb") as f:
                            f.write(data)
                    return True, filename, len(data)
        except Exception:
            time.sleep(0.5)

    return False, filename, 0


def generate_matchsettings_xml(map_filenames: List[str], time_limit_sec: int = 180) -> str:
    """Generates TrackMania MatchSettings playlist XML for arcade/local rotation."""
    lines = [
        '<?xml version="1.0" encoding="utf-8" ?>',
        "<playlist>",
        "  <gameinfos>",
        "    <game_mode>0</game_mode>",
        "    <script_name>Trackmania/TM_TimeAttack_Online</script_name>",
        "    <title>Trackmania</title>",
        "  </gameinfos>",
        "  <mode_script_settings>",
        f'    <setting name="S_TimeLimit" type="integer" value="{time_limit_sec}"/>',
        '    <setting name="S_WarmUpNb" type="integer" value="0"/>',
        '    <setting name="S_ChatTime" type="integer" value="5"/>',
        "  </mode_script_settings>",
        "  <startindex>0</startindex>",
    ]
    for fn in map_filenames:
        lines.append(f"  <map><file>AI_1000/{fn}</file></map>")
    lines.append("</playlist>")
    return "\n".join(lines)


def main():
    tm_user_dir = os.path.expanduser("~/Documents/Trackmania")
    tm_maps_dir = os.path.join(tm_user_dir, "Maps", "AI_1000")
    tm_matchsettings_dir = os.path.join(tm_user_dir, "MatchSettings")
    local_mirror_dir = "D:/Project/tmrl/data/maps_1k"

    os.makedirs(tm_maps_dir, exist_ok=True)
    os.makedirs(tm_matchsettings_dir, exist_ok=True)
    os.makedirs(local_mirror_dir, exist_ok=True)

    print("=================================================================", flush=True)
    print("      TRACKMANIA 2020: 1,000 DIVERSE MAPS HARVEST ENGINE        ", flush=True)
    print("=================================================================", flush=True)
    print(f"Target In-Game Dir : {tm_maps_dir}", flush=True)
    print(f"MatchSettings Dir  : {tm_matchsettings_dir}", flush=True)
    print(f"Project Mirror Dir : {local_mirror_dir}", flush=True)
    print("-----------------------------------------------------------------", flush=True)

    # Step 1: Collect catalog across 10 categories
    catalog = []
    seen_ids: Set[int] = set()

    for cat_name, tag_id in CATEGORIES:
        items = fetch_category_maps(cat_name, tag_id, seen_ids)
        catalog.extend(items)
        time.sleep(0.05)

    print(f"\n[+] Total unique maps cataloged: {len(catalog)}", flush=True)
    output_dirs = [tm_maps_dir, local_mirror_dir]

    # Step 2: Parallel Download
    print("\n[+] Starting parallel download (15 workers)...", flush=True)
    start_time = time.time()
    successful_downloads = []
    total_bytes = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        future_to_map = {
            executor.submit(download_single_map, item, cat, output_dirs): (item, cat)
            for item, cat in catalog
        }

        completed = 0
        total = len(catalog)
        for future in concurrent.futures.as_completed(future_to_map):
            completed += 1
            ok, fn, nbytes = future.result()
            if ok:
                successful_downloads.append(fn)
                total_bytes += nbytes

            if completed % 25 == 0 or completed == total:
                elapsed = time.time() - start_time
                speed_mb = (total_bytes / (1024 * 1024)) / max(0.1, elapsed)
                print(
                    f" Progress: {completed:4d}/{total} maps | "
                    f"Success: {len(successful_downloads):4d} | "
                    f"Saved: {total_bytes / (1024*1024):5.1f} MB | "
                    f"Speed: {speed_mb:4.1f} MB/s | "
                    f"Elapsed: {elapsed:4.1f}s",
                    flush=True,
                )

    # Step 3: Write metadata manifest
    manifest_path = os.path.join(local_mirror_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "count": len(successful_downloads),
                "timestamp": time.time(),
                "categories": [c[0] for c in CATEGORIES],
                "maps": successful_downloads,
            },
            f,
            indent=2,
        )
    print(f"\n[+] Saved manifest: {manifest_path}", flush=True)

    # Step 4: Write MatchSettings playlist XML for TrackMania Arcade Room
    matchsettings_xml = generate_matchsettings_xml(successful_downloads, time_limit_sec=180)
    matchsettings_path = os.path.join(tm_matchsettings_dir, "AI_1000_Arcade.txt")
    with open(matchsettings_path, "w", encoding="utf-8") as f:
        f.write(matchsettings_xml)
    print(f"[+] Created TrackMania Arcade Playlist: {matchsettings_path}", flush=True)

    print("\n=================================================================", flush=True)
    print(f" ALL DONE! Successfully harvested {len(successful_downloads)} diverse maps.", flush=True)
    print(" You can now launch TrackMania -> Local -> Arcade -> AI_1000_Arcade!", flush=True)
    print("=================================================================", flush=True)


if __name__ == "__main__":
    main()
