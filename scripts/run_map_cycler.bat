@echo off
title TMRL Map Cycler Daemon (Starter Access)
cd /d "D:\Project\tmrl"
echo =========================================================
echo   TRACKMANIA 2020 AUTO MAP CYCLER (STARTER ACCESS)
echo   Cycling through 912 AI maps every 3 minutes (180s)
echo =========================================================
echo In Trackmania, make sure you loaded:
echo   Play -> Local -> Play a map -> My Maps -> AI_Current_Track
echo =========================================================
D:\miniconda3\envs\rcdream\python.exe scripts\cycle_map_macro.py --auto --interval 180
pause
