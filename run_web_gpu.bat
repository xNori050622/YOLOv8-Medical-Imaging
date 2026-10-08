@echo off
REM ============================================================
REM  YOLOv8-Medical-Imaging - one-click Streamlit launcher (GPU)
REM  Usage: double-click this file, or run:  run_web_gpu.bat
REM  To change port: streamlit run app.py --server.port 8501
REM
REM  Difference from run_web.bat:
REM    run_web.bat     uses venv_main311 (CPU torch)
REM    run_web_gpu.bat uses venv_gpu     (CUDA torch) -> GPU inference
REM  Measured on this machine (RTX 4060 Laptop, detect best.pt, imgsz 640):
REM    CPU 0.070 s/img   ->   GPU 0.014 s/img   (~5x faster, first call ~2.8 s)
REM ============================================================
setlocal
cd /d "%~dp0"

set "PY=D:\infynova\venv_gpu\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo [INFO] Working dir : %CD%
echo [INFO] Python      : %PY%
echo [INFO] Starting Streamlit ... (press Ctrl+C to stop)
echo.

"%PY%" -m streamlit run app.py
set "RC=%ERRORLEVEL%"
endlocal & exit /b %RC%
