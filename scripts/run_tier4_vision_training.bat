@echo off
title Tier 4 Multi-Track Vision Foundation Trainer (1,000,000 Steps)
cd /d "D:\Project\tmrl"
echo =====================================================================
echo    TIER 4 MULTI-TRACK VISION FOUNDATION TRAINER
echo    Steps: 1,000,000 ^| Frames: 1.02 Billion ^| In-Game: 14,222 Hours
echo    Autosave: Every 2,500 steps ^| Milestones: Every 50,000 steps
echo =====================================================================
D:\miniconda3\envs\rcdream\python.exe scripts/train_multitrack_vision_curriculum.py --steps 1000000 --batch_size 32 --save_every 2500
pause
