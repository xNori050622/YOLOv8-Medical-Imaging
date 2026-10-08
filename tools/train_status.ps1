# ============================================================
#  YOLOv8 detect 训练进度看板   —— 跑一次, 打印当前状态
#  位置: 项目 tools\ 目录 (与 train_detect_gpu.bat 同目录)
#  用法(VS Code 终端):  & .\tools\train_status.ps1
#  用法(CMD / 双击):    tools\train_status.bat
#  用法(菜单):          tools\train_detect_gpu.bat  ->  [7]
# ============================================================
$Runs    = 'D:\YOLOv8-Medical-Imaging\runs\detect\train'
$Csv     = "$Runs\results.csv"
$Weights = "$Runs\weights"
$Logs    = @("$Runs\logs\train_console.log")   # 训练控制台日志(已从 D:\Temp\resume3.log 归档)
$Total   = 100

Write-Host ''
Write-Host '=== YOLOv8 血细胞检测 训练进度 ===' -ForegroundColor Cyan
Write-Host ("  时间    : " + (Get-Date -Format 'yyyy/MM/dd HH:mm:ss'))

# ---------- 1. 进程 ----------
# 只认「命令行里带 train.py」的 python。
# 不能直接数 Get-Process python —— 网页演示(run_web_gpu.bat)同样是 python 进程,
# 会被误判成「正在训练」(实测踩过: 看板显示 epoch 102/100)。
$py = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like '*train.py*' })
$running = $py.Count -gt 0

# ---------- 2. results.csv ----------
$rows = @()
if (Test-Path $Csv) {
    $rows = @(Get-Content $Csv | Select-Object -Skip 1 | Where-Object { $_.Trim() -ne '' })
}
$done = $rows.Count

# 每轮耗时: 用最近 5 轮的时间列差值
$perEpoch = 25.0
if ($done -ge 6) {
    $t1 = [double]($rows[-1].Split(',')[1])
    $t0 = [double]($rows[-6].Split(',')[1])
    if (($t1 - $t0) -gt 0) { $perEpoch = ($t1 - $t0) / 5.0 }
}

if ($running) {
    $left = $Total - $done
    if ($left -lt 0) { $left = 0 }
    if ($done -ge $Total) {
        # 训练进程还没退出, 但轮数已经跑满 —— 别再显示 epoch 102/100
        Write-Host '  状态    : 已跑完 100 轮 (训练进程尚未退出)' -ForegroundColor Green
        Write-Host ("  进度    : epoch {0} / {1}" -f $done, $Total)
    } else {
        Write-Host '  状态    : 正在训练' -ForegroundColor Green
        Write-Host ("  进度    : epoch {0} / {1}   已存 {2} 轮, 剩 {3} 轮 约 {4} 分钟" -f ($done + 1), $Total, $done, $left, [math]::Round($left * $perEpoch / 60.0, 1))
    }
    Write-Host ("  速度    : {0} 秒/轮 ({1} 个 train.py 进程)" -f [math]::Round($perEpoch, 1), $py.Count)
} elseif ($done -ge $Total) {
    Write-Host '  状态    : 已跑完 100 轮, 进程已正常退出' -ForegroundColor Green
} elseif ($done -gt 0) {
    Write-Host ("  状态    : 已停止! 只完成 {0}/{1} 轮" -f $done, $Total) -ForegroundColor Yellow
    Write-Host '            续训: train_detect_gpu.bat 选 [6], 或 train.py detect --resume --workers 2'
} else {
    Write-Host '  状态    : 还没开始 (没有 results.csv)' -ForegroundColor Yellow
}

# ---------- 3. 指标 ----------
if ($done -gt 0) {
    $best50 = -1.0; $best50e = -1; $best95 = -1.0; $best95e = -1
    foreach ($r in $rows) {
        $p = $r.Split(',')
        $a = [double]$p[7]; $b = [double]$p[8]
        if ($a -gt $best50) { $best50 = $a; $best50e = [int]$p[0] }
        if ($b -gt $best95) { $best95 = $b; $best95e = [int]$p[0] }
    }
    Write-Host '  本轮最优: ' -NoNewline
    Write-Host ("mAP50 = {0} (epoch {1})   mAP50-95 = {2} (epoch {3})" -f [math]::Round($best50, 5), $best50e, [math]::Round($best95, 5), $best95e) -ForegroundColor Green
    Write-Host '  基线对比: 原 best.pt 在新 val 上 mAP50 = 0.8974'

    Write-Host '  最近 5 轮:' -NoNewline
    Write-Host ''
    $tail = @($rows | Select-Object -Last 5)
    foreach ($r in $tail) {
        $p = $r.Split(',')
        Write-Host ("    epoch {0,3}   mAP50 = {1}   mAP50-95 = {2}   cls_loss = {3}" -f [int]$p[0], $p[7], $p[8], $p[2])
    }
}

# ---------- 4. 权重文件 ----------
Write-Host '  权重    :'
foreach ($n in @('best.pt', 'last.pt', 'best.pt.bak')) {
    $f = Join-Path $Weights $n
    if (Test-Path $f) {
        $i = Get-Item $f
        $tag = ''
        if ($n -eq 'best.pt.bak') { $tag = '   <- 原始 Colab 权重, 别动' }
        elseif ($i.Length -lt 10MB) { $tag = '   <- 已剥离 optimizer (正常终态)' }
        Write-Host ("    {0,-14} {1,8:N1} MB   {2}{3}" -f $n, ($i.Length / 1MB), $i.LastWriteTime.ToString('MM-dd HH:mm:ss'), $tag)
    }
}

# ---------- 5. 内存 / GPU ----------
$os = Get-CimInstance Win32_OperatingSystem
$cAvail = [math]::Round($os.FreeVirtualMemory / 1MB, 1)
$pAvail = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
$color = 'Gray'
if ($cAvail -lt 4) { $color = 'Yellow' }
if ($cAvail -lt 2) { $color = 'Red' }
Write-Host ("  内存    : 提交可用 {0} GB   物理可用 {1} GB" -f $cAvail, $pAvail) -ForegroundColor $color
if ($cAvail -lt 4) { Write-Host '            (余量偏低: 关掉豆包/QQ/飞书/Edge 可腾出约 5 GB)' -ForegroundColor Yellow }

$gpu = & nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader 2>$null
if ($gpu) { Write-Host ("  GPU     : " + $gpu) }

# ---------- 6. 实时日志 ----------
$log = $null
foreach ($l in $Logs) { if (Test-Path $l) { $log = $l; break } }
if ($log) {
    Write-Host ("  日志    : " + $log)
    $bar = @(Get-Content $log -Encoding UTF8 -ErrorAction SilentlyContinue | Where-Object { $_ -match '/108' })
    if ($bar.Count -gt 0 -and $running) {
        $line = $bar[-1] -replace '\s+', ' '
        Write-Host ("  当前轮  :" + $line.Trim())
    }
    $mem = @(Get-Content $log -Encoding UTF8 -ErrorAction SilentlyContinue | Select-String -SimpleMatch '[mem]')
    if ($mem.Count -gt 0 -and $running) {
        $mline = ($mem[-1].Line -replace '.*(\[mem\].*)$', '$1').Trim()
        Write-Host ("  " + $mline)
    }

    $errLog = $log -replace '\.log$', '.err.log'
    if (Test-Path $errLog) {
        $errs = @(Get-Content $errLog -Encoding UTF8 -ErrorAction SilentlyContinue)
        if ($errs.Count -eq 0) {
            Write-Host '  错误    : 无 (err.log 是空的)' -ForegroundColor Green
        } else {
            Write-Host ("  错误    : err.log 有 {0} 行, 末尾内容:" -f $errs.Count) -ForegroundColor Red
            $errs | Select-Object -Last 5 | ForEach-Object { Write-Host ("            " + $_) -ForegroundColor Red }
        }
    }
}
Write-Host ''

