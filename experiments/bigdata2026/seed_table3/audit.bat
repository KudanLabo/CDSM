@echo off
REM Step 1: audit existing Table III runs (read only) and export the seed42 configs.
REM Run this on the machine that holds the Table III outputs (outputs\ or outputs_local\).
chcp 65001 > nul
setlocal
cd /d "%~dp0"
set CDSM_REPO=C:\Users\ando\CDSM
if not "%~1"=="" set CDSM_REPO=%~1
set PYTHONIOENCODING=utf-8
if not exist logs mkdir logs
python seed_experiment.py audit --repo "%CDSM_REPO%" --metrics --tag %COMPUTERNAME% > "logs\audit_%COMPUTERNAME%.log" 2>&1
python seed_experiment.py export --repo "%CDSM_REPO%" >> "logs\audit_%COMPUTERNAME%.log" 2>&1
type "logs\audit_%COMPUTERNAME%.log"
endlocal
