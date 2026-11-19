@echo off
REM Independence driver B (Quadro P620). 4-GPU assignment 2026-11-18:
REM   seed 12 FX29, Industry, iShares -> seed 42 iShares -> seed 22 iShares
REM Protocol: epoch validation 3 MC x 300 steps, final sampling 3 repeats.
chcp 65001 > nul
setlocal
cd /d "%~dp0"
set CDSM_REPO=C:\Users\ando\CDSM
set DATA_ROOT=%USERPROFILE%\.gluonts\datasets
set PYTHONIOENCODING=utf-8
set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
set CUDA_DEVICE_ORDER=FASTEST_FIRST
set CUDA_VISIBLE_DEVICES=1
set PY=C:\Users\ando\.venvs\cfdiff\Scripts\python.exe
set OPTS=--repo "%CDSM_REPO%" --machine independence --data-root "%DATA_ROOT%" --epoch-mc 3 --epoch-steps 300 --sample-repeats 3
set LOG=logs\driver_independence_quadro.log
"%PY%" seed_experiment.py run %OPTS% --seeds 12 --datasets fx29 industry ishares >> "%LOG%" 2>&1
"%PY%" seed_experiment.py run %OPTS% --seeds 42 --datasets ishares >> "%LOG%" 2>&1
"%PY%" seed_experiment.py run %OPTS% --seeds 22 --datasets ishares >> "%LOG%" 2>&1
echo ALL_DONE %date% %time% >> "%LOG%"
endlocal
