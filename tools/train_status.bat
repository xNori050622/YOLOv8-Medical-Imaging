@echo off
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0train_status.ps1"
if /i not "%~1"=="/nopause" pause
