@echo off
REM ---- go to your repo ----
cd /d C:\Users\gd3470\Desktop\ssl\utils

REM ---- run training & log ----
set LOGDIR=C:\Users\gd3470\Desktop\ssl\artifacts\humaid\logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
"C:\Users\gd3470\micromamba\envs\train\python.exe" -u ./run_humaid.py > "%LOGDIR%\train_%DATE:~-4%%DATE:~4,2%%DATE:~7,2%_%TIME:~0,2%%TIME:~3,2%.log" 2>&1
