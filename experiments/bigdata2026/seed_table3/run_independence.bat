@echo off
REM Table III seed experiment: Independence part. Edit CDSM_REPO below before the first run.
REM Usage:  run_independence.bat               (full run, resumes where it stopped)
REM         run_independence.bat --dry-run     (print commands only)
chcp 65001 > nul
setlocal
cd /d "%~dp0"
REM ---- edit here: CDSM repository root on this machine (folder that contains src\ and outputs\)
set CDSM_REPO=C:\Users\ando\CDSM
set DATA_ROOT=%USERPROFILE%\.gluonts\datasets
set PYTHONIOENCODING=utf-8
REM torch>=2.6: our own checkpoints contain omegaconf objects (needed for training and sampling)
set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
REM Independence has no conda cfdiff env; use the venv built for this experiment
set PY=C:\Users\ando\.venvs\cfdiff\Scripts\python.exe
"%PY%" seed_experiment.py run --repo "%CDSM_REPO%" --machine independence --data-root "%DATA_ROOT%" %* >> "logs\driver_independence.log" 2>&1
type "logs\driver_independence.log"
endlocal
