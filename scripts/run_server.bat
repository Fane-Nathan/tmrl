@echo off
title TMRL Central Server
cd /d "D:\Project\tmrl"
echo ===================================================
echo             TMRL CENTRAL RELAY SERVER
echo ===================================================
D:\miniconda3\envs\rcdream\python.exe -m tmrl --server
pause
