@echo off
REM Table III seed experiment: Audacity part. Edit CDSM_REPO below before the first run.
REM Usage:  run_audacity.bat               (full run, resumes where it stopped)
REM         run_audacity.bat --dry-run     (print commands only)
chcp 65001 > nul
setlocal
cd /d "%~dp0"
REM ---- edit here: CDSM repository root on this machine (folder that contains src\ and outputs\)
set CDSM_REPO=C:\Users\ando\CDSM
set DATA_ROOT=%USERPROFILE%\.gluonts\datasets
set PYTHONIOENCODING=utf-8
REM torch >= 2.6: checkpoints contain omegaconf objects, so allow full unpickling
set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
REM one GPU per job; PCI order: 0 = GTX 1050 Ti (laptop), 1 = GTX 980 Ti. Override by setting CUDA_VISIBLE_DEVICES before calling.
set CUDA_DEVICE_ORDER=PCI_BUS_ID
if not defined CUDA_VISIBLE_DEVICES set CUDA_VISIBLE_DEVICES=1
REM Audacity uses the venv built for this experiment
set PY=C:\Users\ando\.venvs\cfdiff\Scripts\python.exe
"%PY%" seed_experiment.py run --repo "%CDSM_REPO%" --machine audacity --data-root "%DATA_ROOT%" %* >> "logs\driver_audacity.log" 2>&1
type "logs\driver_audacity.log"
endlocal
