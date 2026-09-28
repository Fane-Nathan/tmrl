@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
"D:\chatgpt\chatgpt_api\.venv\Scripts\python.exe" "D:\chatgpt\chatgpt_api\codebase_chat.py" --root "%~dp0." %*
