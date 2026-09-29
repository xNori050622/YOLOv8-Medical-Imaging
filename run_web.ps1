# ============================================================
#  YOLOv8-Medical-Imaging - one-click Streamlit launcher (PowerShell)
#  Usage:  powershell -ExecutionPolicy Bypass -File .\run_web.ps1
#  To change port: streamlit run app.py --server.port 8501
# ============================================================
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$py = 'D:\infynova\venv_main311\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $py)) { $py = 'python' }

Write-Host "[INFO] Working dir : $($PWD.Path)"
Write-Host "[INFO] Python      : $py"
Write-Host "[INFO] Starting Streamlit ... (press Ctrl+C to stop)" -ForegroundColor Cyan

& $py -m streamlit run app.py
