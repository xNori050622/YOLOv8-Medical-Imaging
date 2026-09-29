@echo off
REM ============================================================
REM  YOLOv8-Medical-Imaging - one-click Streamlit launcher
REM  Usage: double-click this file, or run:  run_web.bat
REM  To change port: streamlit run app.py --server.port 8501
REM ============================================================
setlocal
cd /d "%~dp0"

set "PY=D:\infynova\venv_main311\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo [INFO] Working dir : %CD%
echo [INFO] Python      : %PY%
echo [INFO] Starting Streamlit ... (press Ctrl+C to stop)
echo.

"%PY%" -m streamlit run app.py
set "RC=%ERRORLEVEL%"
endlocal & exit /b %RC%
