@echo off
rem ===================================================================
rem  YOLOv8-Medical-Imaging / detection(RBC-WBC-Platelets) GPU 训练助手
rem  用法: 直接双击 = 菜单；也可带参数一次性执行:
rem      train_detect_gpu.bat check   自检(权重+数据集)
rem      train_detect_gpu.bat eval    评估现有 best.pt (valid 集, 存 JSON)
rem      train_detect_gpu.bat smoke   冒烟训练 1 epoch, 输出到 runs\detect\smoke
rem      train_detect_gpu.bat train   正式训练 100 epoch, 输出到 runs\detect\train
rem      train_detect_gpu.bat resume  续训：从 last.pt 继续跑（崩溃/断电后接上）
rem      train_detect_gpu.bat web     启动 Streamlit 网页 (GPU)
rem  位置: 本文件与 train_status.ps1 同在项目 tools\ 目录下, 用 %~dp0 互相引用。
rem ===================================================================
chcp 65001 >nul
setlocal
set "PROJECT=D:\YOLOv8-Medical-Imaging"
set "PY=D:\infynova\venv_gpu\Scripts\python.exe"

if not exist "%PY%" (
  echo [ERROR] 找不到 GPU 环境: %PY%
  echo         请检查该 venv 是否存在, 里面有 CUDA 版 torch。
  pause & exit /b 1
)
cd /d "%PROJECT%" || (echo [ERROR] 找不到项目目录 "%PROJECT%" & pause & exit /b 1)

if /i "%~1"=="check" goto :check
if /i "%~1"=="eval"  goto :eval
if /i "%~1"=="smoke" goto :smoke
if /i "%~1"=="train" goto :train
if /i "%~1"=="resume" goto :resume
if /i "%~1"=="web"   goto :web

:menu
echo ============================================================
echo  YOLOv8-Medical-Imaging - 血细胞目标检测 (detect)
echo  项目: %PROJECT%
echo  环境: %PY%
echo ============================================================
echo   [1] 自检 (权重 + 数据集)                    秒级
echo   [2] 评估现有 best.pt (valid 集, 存 JSON)    几十秒
echo   [3] 冒烟训练 1 epoch  (不动现有 best.pt)    几分钟
echo   [4] 正式训练 100 epoch (会备份 best.pt.bak) 约 35 分钟
echo   [5] 启动网页演示 run_web_gpu.bat (GPU 加速)
echo   [6] 续训 (从 last.pt 接上, 崩溃后用它)       剩余轮数 x 21 秒
echo   [7] 查看训练进度看板 (跑在哪一轮/指标/内存)  秒级
echo   [0] 退出
echo ============================================================
set /p CH=请选择 [1/2/3/4/5/6/7/0]:
if "%CH%"=="1" goto :check
if "%CH%"=="2" goto :eval
if "%CH%"=="3" goto :smoke
if "%CH%"=="4" goto :train
if "%CH%"=="5" goto :web
if "%CH%"=="6" goto :resume
if "%CH%"=="7" goto :status
if "%CH%"=="0" exit /b 0
echo 无效选择 & pause & goto :menu

:check
echo.
echo [1] 自检中...
"%PY%" train.py check
goto :done

:eval
echo.
echo [2] 评估现有 best.pt (split=valid, device=0)...
"%PY%" evaluate.py detect --split valid --device 0 --save runs\eval_detect_valid.json
goto :done

:smoke
echo.
echo [3] 冒烟训练: 1 epoch, batch 16, 输出到 runs\detect\smoke
echo     workers 也用 2: 冒烟只为验证流程通不通, 没必要冒内存不足的风险。
echo     --no-amp 可避免 ultralytics 为 AMP 自检联网下载 yolo26n.pt
"%PY%" train.py detect --epochs 1 --device 0 --name smoke --no-amp --workers 2
goto :done

:train
echo.
echo [4] 正式训练: 100 epoch, 输出到 runs\detect\train
echo     注意: 现有 runs\detect\train\weights\best.pt 会先被备份为 best.pt.bak
echo     实测: workers=2 约 5.6 it/s (19-22 秒/epoch), 100 轮约 35 分钟, 显存 3.9G/8.2G。
echo     workers 为什么只能用 2: 本机 16G 内存, 而"提交内存"(commit) 上限约 35.5G,
echo     其中空闲基线就占 22.8G (VS Code 3.6G + DTS音频1.6G + 豆包1.5G + QQ1.3G
echo     + 飞书1.2G + Edge1.5G + 微信0.5G + MySQL/SQLServer 1.3G 等),
echo     训练主进程还要占 9.3G, 只剩约 3.4G 给 worker, 而每个 worker 进程约 1.2G。
echo     workers 给多了页文件会被撑大, 物理内存见底, worker 里 cv2 连 1.35MB
echo     都申请不到 (cv2.error: Insufficient memory), 整场崩 (实测 workers=8 第12轮崩)。
echo     [!] 训练期间把豆包/QQ/飞书/Edge/微信关掉, 能腾出约 5G, 跑得最舒服。 [!]
echo     日志里 [mem] 行每 60 秒报一次余量, 可用低于 4G 会告警。
echo     checkpoint 保存已加固 (内存 buffer 不重试 + 整次重试),
echo     万一还是崩了/断电了, 用菜单 [6] 或 --resume 从 last.pt 接上, 不用从头再来。
echo     建议插好电源、保证散热。若卡在扫描数据集不动, 改 --workers 0 重跑。
set /p OK=确认开始? [y/N]:
if /i not "%OK%"=="y" goto :cancelled
"%PY%" train.py detect --device 0 --no-amp --workers 2
goto :done

:resume
echo.
echo [6] 续训: 从 runs\detect\train\weights\last.pt 继续跑到 100 epoch
echo     epochs/data/batch/amp 等沿用 checkpoint, 但 workers 这里强制用 2,
echo     覆盖 checkpoint 里的 8 —— 8 会起 16+ 个常驻 worker, 顶满提交内存后
echo     worker 里 cv2 连 1.35MB 都申请不到, 整场崩 (实测第 12 轮)。
if not exist "%PROJECT%\runs\detect\train\weights\last.pt" (
  echo.
  echo [ERROR] 找不到 last.pt, 没有可续训的训练。
  goto :done
)
"%PY%" train.py detect --resume --device 0 --workers 2
goto :done

:status
echo.
echo [7] 训练进度看板...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0train_status.ps1"
goto :done

:web
echo.
echo [5] 启动网页 (GPU 版, 关闭窗口即停止)...
if exist "%PROJECT%\run_web_gpu.bat" (
  call "%PROJECT%\run_web_gpu.bat"
) else (
  echo     没找到 run_web_gpu.bat, 回退到 CPU 版 run_web.bat
  call "%PROJECT%\run_web.bat"
)
goto :done

:cancelled
echo 已取消。

:done
echo.
echo ============================================================
echo  执行结束。输出目录: %PROJECT%\runs
echo ============================================================
pause
exit /b 0
