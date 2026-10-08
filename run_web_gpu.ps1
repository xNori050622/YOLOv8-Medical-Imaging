# ============================================================
#  YOLOv8-Medical-Imaging - one-click Streamlit launcher (GPU)
#  Usage:  powershell -ExecutionPolicy Bypass -File .\run_web_gpu.ps1
#  To change port: streamlit run app.py --server.port 8501
#
#  Difference from run_web.ps1:
#    run_web.ps1     uses venv_main311 (CPU torch)
#    run_web_gpu.ps1 uses venv_gpu     (CUDA torch) -> GPU inference
#  Measured on this machine (RTX 4060 Laptop, detect best.pt, imgsz 640):
#    CPU 0.070 s/img   ->   GPU 0.014 s/img   (~5x faster, first call ~2.8 s)
# ============================================================
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$py = 'D:\infynova\venv_gpu\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $py)) { $py = 'python' }

Write-Host "[INFO] Working dir : $($PWD.Path)"
Write-Host "[INFO] Python      : $py"
Write-Host "[INFO] Starting Streamlit ... (press Ctrl+C to stop)" -ForegroundColor Cyan

& $py -m streamlit run app.py
