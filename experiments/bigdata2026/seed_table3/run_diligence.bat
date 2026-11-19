@echo off
REM Table III seed experiment: Diligence part. Edit CDSM_REPO below before the first run.
REM Usage:  run_diligence.bat               (full run, resumes where it stopped)
REM         run_diligence.bat --dry-run     (print commands only)
chcp 65001 > nul
setlocal
cd /d "%~dp0"
REM ---- edit here: CDSM repository root on this machine (folder that contains src\ and outputs\)
set CDSM_REPO=C:\Users\ando\CDSM
set DATA_ROOT=%USERPROFILE%\.gluonts\datasets
set PYTHONIOENCODING=utf-8
REM torch >= 2.6: checkpoint contains omegaconf objects, so disable weights_only loading
set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
"C:\Users\ando\CDSM\.venv\Scripts\python.exe" seed_experiment.py run --repo "%CDSM_REPO%" --machine diligence --data-root "%DATA_ROOT%" %* >> "logs\driver_diligence.log" 2>&1
type "logs\driver_diligence.log"
endlocal
