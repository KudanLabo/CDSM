@echo off
REM DCC-GARCH long-window experiment: all four datasets, all lookbacks, fixed and MLE (a,b).
REM Usage:  run_all.bat            (uses %USERPROFILE%\.gluonts\datasets)
REM         run_all.bat D:\data    (another data root)
chcp 65001 > nul
setlocal
cd /d "%~dp0"
set DATA_ROOT=%USERPROFILE%\.gluonts\datasets
if not "%~1"=="" set DATA_ROOT=%~1
python dcc_long_window.py --data-root "%DATA_ROOT%" --dataset exchange fx29 industry49 ishares14 --lookbacks context 250 500 expanding --ab both > "run_%COMPUTERNAME%.log" 2>&1
type "run_%COMPUTERNAME%.log"
endlocal
