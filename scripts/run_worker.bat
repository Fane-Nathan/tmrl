@echo off
title TMRL Foundation Rollout Worker
cd /d "D:\Project\tmrl"
echo ===================================================
echo   TMRL FOUNDATION ROLLOUT WORKER (20 Hz RTGYM)
echo ===================================================
echo Make sure TrackMania is open and running on the target track.
D:\miniconda3\envs\rcdream\python.exe -m tmrl --worker
pause
