@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo Starting PARKVISION AI on http://localhost:8000/login
python -m uvicorn src.api_server:app --host 0.0.0.0 --port 8000
