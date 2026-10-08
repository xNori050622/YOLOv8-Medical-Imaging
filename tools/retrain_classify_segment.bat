@echo off
rem ===================================================================
rem  YOLOv8-Medical-Imaging / 分类 + 分割 重训助手 (classify / segment)
rem  对应 README 章节「5. Re-training classification and segmentation」的 9 个步骤。
rem  detect 的事请走 tools\train_detect_gpu.bat；本文件只管 cls / seg 这两件事。
rem  用法: 直接双击 = 菜单；也可带参数一次性执行:
rem      retrain_classify_segment.bat check     步骤1 自检(基础权重 + 三个数据集)
rem      retrain_classify_segment.bat prepare   步骤3 把 BUSI 掩码转多边形并划分
rem      retrain_classify_segment.bat split     步骤4 从 train/ 切出真正的 val/ (201/50/66)
rem      retrain_classify_segment.bat unval     步骤4 的反向: 按清单把 val/ 搬回 train/
rem      retrain_classify_segment.bat archive   步骤5 归档上游 checkpoint —— 训练覆盖前做
rem      retrain_classify_segment.bat smoke     冒烟: classify + segment 各 1 epoch
rem      retrain_classify_segment.bat classify  步骤6 训练分类 100 epoch
rem      retrain_classify_segment.bat segment   步骤6 训练分割 100 epoch
rem      retrain_classify_segment.bat resume    续训(从 last.pt 接上, 崩溃/断电后用)
rem      retrain_classify_segment.bat eval      步骤7 重录 runs\eval_*.json
rem      retrain_classify_segment.bat guard     步骤8 校验 README 自述统计
rem      retrain_classify_segment.bat all       按 1 - 3 - 4 - 6 - 7 - 8 顺序连跑
rem  位置: 本文件与 train_detect_gpu.bat / train_status.ps1 同在项目 tools\ 目录下。
rem ===================================================================
chcp 65001 >nul
setlocal
set "RESULT=0"
set "PROJECT=D:\YOLOv8-Medical-Imaging"
set "PY=D:\infynova\venv_gpu\Scripts\python.exe"

if not exist "%PY%" (
  echo [ERROR] 找不到 GPU 环境: %PY%
  echo         重训需要 ultralytics + torch; 换机器请先改本文件顶部的 PY 变量。
  pause & exit /b 1
)
cd /d "%PROJECT%" || (echo [ERROR] 找不到项目目录 "%PROJECT%" & pause & exit /b 1)

if /i "%~1"=="check"    goto :check
if /i "%~1"=="prepare"  goto :prepare
if /i "%~1"=="split"    goto :split
if /i "%~1"=="unval"    goto :unval
if /i "%~1"=="archive"  goto :archive
if /i "%~1"=="smoke"    goto :smoke
if /i "%~1"=="classify" goto :classify
if /i "%~1"=="segment"  goto :segment
if /i "%~1"=="resume"   goto :resume
if /i "%~1"=="eval"     goto :eval
if /i "%~1"=="guard"    goto :guard
if /i "%~1"=="all"      goto :all

:menu
echo ============================================================
echo  YOLOv8-Medical-Imaging - 分类 / 分割 重训
echo  项目: %PROJECT%
echo  环境: %PY%
echo ============================================================
echo   [1] 自检: 基础权重 + 三个数据集              (步骤 1)  秒级
echo   [2] 准备 BUSI 分割数据集 (掩码转多边形)      (步骤 3)  约 1 分钟
echo   [3] 切出真正的 val/ 给分类用                 (步骤 4)  秒级, 移动 50 个文件
echo   [4] 归档上游 checkpoint (训练覆盖前!)        (步骤 5)  秒级
echo   [5] 冒烟训练: cls + seg 各 1 epoch                     几分钟, 不动现有 best.pt
echo   [6] 正式训练分类 100 epoch                   (步骤 6)
echo   [7] 正式训练分割 100 epoch                   (步骤 6)
echo   [8] 重录指标 runs\eval_*.json                (步骤 7)
echo   [9] 校验 README 自述统计                     (步骤 8)
echo  [10] 撤销分类的 val/ 切分                    (步骤 4 的反向)
echo  [11] 续训 (分类或分割, 从 last.pt 接上)
echo   [0] 退出
echo ============================================================
set /p CH=请选择 [1-9/10/11/0]:
if "%CH%"=="1"  goto :check
if "%CH%"=="2"  goto :prepare
if "%CH%"=="3"  goto :split
if "%CH%"=="4"  goto :archive
if "%CH%"=="5"  goto :smoke
if "%CH%"=="6"  goto :classify
if "%CH%"=="7"  goto :segment
if "%CH%"=="8"  goto :eval
if "%CH%"=="9"  goto :guard
if "%CH%"=="10" goto :unval
if "%CH%"=="11" goto :resume
if "%CH%"=="0"  exit /b 0
echo 无效选择 & pause & goto :menu

rem ---------- 菜单项只是「调用子过程 + 收尾」, 子过程也供 :all 复用 ----------
:check
call :do_check
if errorlevel 1 goto :fail
goto :done

:prepare
call :do_prepare
if errorlevel 1 goto :fail
goto :done

:split
call :do_split
if errorlevel 1 goto :fail
goto :done

:unval
call :do_unval
if errorlevel 1 goto :fail
goto :done

:archive
call :do_archive
if errorlevel 1 goto :fail
goto :done

:smoke
call :do_smoke
if errorlevel 1 goto :fail
goto :done

:classify
call :do_classify
if errorlevel 1 goto :fail
goto :done

:segment
call :do_segment
if errorlevel 1 goto :fail
goto :done

:resume
call :do_resume
if errorlevel 1 goto :fail
goto :done

:eval
call :do_eval
if errorlevel 1 goto :fail
goto :done

:guard
call :do_guard
if errorlevel 1 goto :fail
goto :done

:all
echo.
echo [连跑] 将按顺序执行: 自检 - 准备分割数据 - 切分 val - 训练分类 - 训练分割 - 重录指标 - 校验
echo        不含步骤 5 归档: 「把上游权重留在仓库里」是个需要你本人拍板的决定,
echo        请先单独跑 [4] 或改用 --name train1 (README 步骤 5 讲了两种代价)。
set /p OK=确认开始? [y/N]:
if /i not "%OK%"=="y" goto :cancelled
call :do_check    || goto :fail
call :do_prepare  || goto :fail
call :do_split    || goto :fail
call :do_classify || goto :fail
call :do_segment  || goto :fail
call :do_eval     || goto :fail
call :do_guard    || goto :fail
goto :done


rem ----------------------------------------------------------- 子过程
:do_check
echo.
echo [步骤 1] 自检: 基础权重 + 三个数据集...
"%PY%" train.py check
exit /b %ERRORLEVEL%

:do_prepare
echo.
echo [步骤 3] 准备 BUSI 分割数据集: 掩码转多边形 + 按病例名 8:1:1 划分...
"%PY%" train.py segment --prepare
exit /b %ERRORLEVEL%

:do_split
echo.
echo [步骤 4] 从分类数据集的 train/ 里分层切出 20%% 的 val/。
echo          这一步会移动 50 个文件 (切完 train/val/test = 201/50/66),
echo          并删掉过期的 train.cache / val.cache / test.cache, 记录写进
echo          val_split_manifest.json, 所以可以用 [10] 精确撤销。
echo          如果前面已经切过一次 (val/ 非空, 预览会直接拒绝), 那就别再切了,
echo          直接跳到 [4] 归档; 想换比例重切就先跑 [10] 撤销, 或手工给 --force。
echo          先看预览(不移动任何文件):
"%PY%" tools\make_classify_val.py --dry-run
if errorlevel 1 exit /b 1
set /p OK=确认执行切分? [y/N]:
if /i not "%OK%"=="y" goto :do_split_cancel
"%PY%" tools\make_classify_val.py
exit /b %ERRORLEVEL%
:do_split_cancel
echo 已取消, 未移动任何文件。
exit /b 1

:do_unval
echo.
echo [撤销] 按 val_split_manifest.json 把 val/ 里的文件搬回 train/, 并删除清单与缓存。
set /p OK=确认撤销? [y/N]:
if /i not "%OK%"=="y" goto :do_unval_cancel
"%PY%" tools\make_classify_val.py --undo
exit /b %ERRORLEVEL%
:do_unval_cancel
echo 已取消。
exit /b 1

:do_archive
echo.
echo [步骤 5] 归档上游 checkpoint, 必须在「训练」之前做。
echo          runs\classify\train\weights\best.pt 与 runs\segment\train\weights\best.pt
echo          现在都还是上游作者在 Colab 训的原始文件; 默认名训练会把它们覆盖掉,
echo          所以先把字节级副本放进 _archive\ (detect 当初就是这么做的)。
echo          副作用: README「Credits and licensing」里那句「哪个 checkpoint 属于谁」
echo          会因此变成假话, 请按 README 步骤 8 在同一个提交里改掉。
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set "TODAY=%%i"
echo          归档文件名: best_original_colab_%TODAY%.pt
set /p OK=确认归档? [y/N]:
if /i not "%OK%"=="y" goto :do_archive_cancel
if not exist "runs\classify\_archive" mkdir "runs\classify\_archive"
if not exist "runs\segment\_archive" mkdir "runs\segment\_archive"
copy /y "runs\classify\train\weights\best.pt" "runs\classify\_archive\best_original_colab_%TODAY%.pt"
if errorlevel 1 exit /b 1
copy /y "runs\segment\train\weights\best.pt" "runs\segment\_archive\best_original_colab_%TODAY%.pt"
if errorlevel 1 exit /b 1
echo.
echo          归档完成。接下来记得: git add 这两个副本, 并同步 README 的 Credits 段落。
exit /b 0
:do_archive_cancel
echo 已取消, 未复制任何文件。
exit /b 1

:do_smoke
echo.
echo [冒烟] classify + segment 各跑 1 epoch, 输出到 runs\^<task^>\smoke\ (已 gitignore)。
echo        用 --name smoke 就是为了不动 runs\^<task^>\train\weights\best.pt。
echo        --no-amp 可避免 ultralytics 为 AMP 自检联网下载 yolo26n.pt。
"%PY%" train.py classify --epochs 1 --device 0 --name smoke --no-amp --workers 2
if errorlevel 1 exit /b 1
if not exist "segmentation\data.yaml" (
  echo.
  echo [跳过] 没有 segmentation\data.yaml, 分割冒烟跑不了, 先跑菜单 [2] 准备数据集。
  exit /b 1
)
"%PY%" train.py segment --epochs 1 --device 0 --name smoke --no-amp --workers 2
exit /b %ERRORLEVEL%

:do_classify
echo.
echo [步骤 6] 训练分类: 100 epoch, imgsz 224, batch 16, device 0, workers 2。
echo          输出到 runs\classify\train\, 应用与 evaluate.py 读的就是这个目录。
echo          现有 best.pt 会先被备份成 best.pt.bak, 但那是 gitignore 的本地便利副本;
echo          要留在仓库里的东西请先跑 [4] 归档。
echo          训练期间可以另开窗口看进度, 或看 runs\classify\train\results.csv;
echo          tools\train_status.ps1 只盯着 detect, 不跟踪这两个任务。
echo          中途崩了/断电了: 用 [11] 续训, 从 last.pt 接上。
set /p OK=确认开始? [y/N]:
if /i not "%OK%"=="y" goto :do_classify_cancel
"%PY%" train.py classify --device 0 --no-amp --workers 2
exit /b %ERRORLEVEL%
:do_classify_cancel
echo 已取消。
exit /b 1

:do_segment
echo.
echo [步骤 6] 训练分割: 100 epoch, imgsz 640, batch 16, device 0, workers 2。
if not exist "segmentation\data.yaml" (
  echo.
  echo [ERROR] 找不到 segmentation\data.yaml, 先跑菜单 [2] 准备 BUSI 数据集。
  exit /b 1
)
echo          输出到 runs\segment\train\, 现有 best.pt 同样先备份成 best.pt.bak。
set /p OK=确认开始? [y/N]:
if /i not "%OK%"=="y" goto :do_segment_cancel
"%PY%" train.py segment --device 0 --no-amp --workers 2
exit /b %ERRORLEVEL%
:do_segment_cancel
echo 已取消。
exit /b 1

:do_resume
echo.
echo [续训] epochs / data / batch / imgsz 一律沿用 checkpoint 里的记录, 这里只补 device。
echo        c = 分类 (runs\classify\train) , s = 分割 (runs\segment\train)
set /p WH=续训哪个? [c/s]:
if /i "%WH%"=="c" goto :do_resume_cls
if /i "%WH%"=="s" goto :do_resume_seg
echo 无效选择。
exit /b 1
:do_resume_cls
if not exist "runs\classify\train\weights\last.pt" (
  echo [ERROR] 没有 runs\classify\train\weights\last.pt, 没有可续训的分类训练。
  exit /b 1
)
"%PY%" train.py classify --resume --device 0 --workers 2
exit /b %ERRORLEVEL%
:do_resume_seg
if not exist "runs\segment\train\weights\last.pt" (
  echo [ERROR] 没有 runs\segment\train\weights\last.pt, 没有可续训的分割训练。
  exit /b 1
)
"%PY%" train.py segment --resume --device 0 --workers 2
exit /b %ERRORLEVEL%

:do_eval
echo.
echo [步骤 7] 重录指标: 覆盖下面三个 JSON, 旧值留在 git 历史里。
echo          文件名必须是 runs\eval_*.json, 因为 .gitignore 只放行这一种
echo          (runs/**/*.json 其余全部忽略), 换个名字写出来的文件对读者等于不存在。
"%PY%" evaluate.py classify --split val  --save runs\eval_classify_val.json
if errorlevel 1 exit /b 1
"%PY%" evaluate.py classify --split test --save runs\eval_classify_test.json
if errorlevel 1 exit /b 1
"%PY%" evaluate.py segment  --split val  --cross-check --save runs\eval_segment_val.json
exit /b %ERRORLEVEL%

:do_guard
echo.
echo [步骤 8] 校验 README 自述统计 (与 CI 跑的是同一条命令)...
"%PY%" tools\check_repo_consistency.py --strict
echo.
echo          若它不一致, 会打印一行可直接粘贴的替换文本; 顺带 git status --short 只应
echo          出现 README、两个 _archive 副本、runs\eval_*.json 这几个文件。
exit /b %ERRORLEVEL%

rem ----------------------------------------------------------- 收尾
:cancelled
echo 已取消。

:fail
set "RESULT=1"
echo.
echo [中止] 这一步失败或被取消。
if /i "%~1"=="all" echo         连跑模式下, 后面的步骤不会再执行。
goto :done

:done
echo.
echo ============================================================
echo  执行结束。输出目录: %PROJECT%\runs
echo  README 的「5. Re-training classification and segmentation」列出了这一步之后
echo  还要改哪几个数字 (步骤 8), 以及怎么整体回退 (步骤 9)。
echo ============================================================
pause
exit /b %RESULT%
