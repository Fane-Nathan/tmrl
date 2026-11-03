@echo off
title TMRL Dreamer Foundation 1M Master Launcher
cd /d "D:\Project\tmrl"
echo ===================================================
echo   LAUNCHING DISTRIBUTED TMRL DREAMER FOUNDATION 1M
echo ===================================================
echo Starting Server...
start "TMRL Server" cmd /k "scripts\run_server.bat"
timeout /t 3 /nobreak >nul

echo Starting Trainer...
start "TMRL Trainer" cmd /k "scripts\run_trainer.bat"
timeout /t 3 /nobreak >nul

echo Starting Worker...
start "TMRL Worker" cmd /k "scripts\run_worker.bat"

echo All 3 TMRL components launched in separate windows!
pause
