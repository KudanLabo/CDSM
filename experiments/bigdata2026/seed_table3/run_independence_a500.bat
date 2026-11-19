@echo off
REM Independence driver A (RTX A500). 4-GPU assignment 2026-11-18:
REM   seed 42+12 Exchange -> seed 42 FX29 -> seed 22 Exchange
REM Protocol: epoch validation 3 MC x 300 steps, final sampling 3 repeats.
chcp 65001 > nul
setlocal
cd /d "%~dp0"
set CDSM_REPO=C:\Users\ando\CDSM
set DATA_ROOT=%USERPROFILE%\.gluonts\datasets
set PYTHONIOENCODING=utf-8
set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
set CUDA_DEVICE_ORDER=FASTEST_FIRST
set CUDA_VISIBLE_DEVICES=0
set PY=C:\Users\ando\.venvs\cfdiff\Scripts\python.exe
set OPTS=--repo "%CDSM_REPO%" --machine independence --data-root "%DATA_ROOT%" --epoch-mc 3 --epoch-steps 300 --sample-repeats 3
set LOG=logs\driver_independence_a500.log
"%PY%" seed_experiment.py run %OPTS% --seeds 42 12 --datasets exchange >> "%LOG%" 2>&1
"%PY%" seed_experiment.py run %OPTS% --seeds 42 --datasets fx29 >> "%LOG%" 2>&1
"%PY%" seed_experiment.py run %OPTS% --seeds 22 --datasets exchange >> "%LOG%" 2>&1
echo ALL_DONE %date% %time% >> "%LOG%"
endlocal
