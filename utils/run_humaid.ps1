# ---- Config ----
$repo    = "C:\Users\gd3470\Desktop\ssl\utils"
$python  = "C:\Users\gd3470\micromamba\envs\train\python.exe"
$script  = "run_humaid.py"
$logDir  = "C:\Users\gd3470\Desktop\ssl\artifacts\humaid\logs"

# ---- Setup ----
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

# Timestamp + file paths
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$currJob   = Join-Path $logDir "train_$timestamp"
New-Item -ItemType Directory -Path $currJob | Out-Null

$logFile   = Join-Path $currJob "train_$timestamp.log"
$errFile   = Join-Path $currJob "train_$timestamp.err"
$pidFile   = Join-Path $currJob "train_$timestamp.pid"

# ---- Move to repo and run detached ----
Set-Location $repo
$proc = Start-Process $python `
    -ArgumentList "-u", $script `
    -RedirectStandardOutput $logFile `
    -RedirectStandardError $errFile `
    -WindowStyle Hidden `
    -PassThru

# Save PID for later killing
$proc.Id | Out-File $pidFile -Encoding ascii