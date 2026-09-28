$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& "D:\chatgpt\chatgpt_api\.venv\Scripts\python.exe" "D:\chatgpt\chatgpt_api\codebase_chat.py" --root $scriptDir @args
