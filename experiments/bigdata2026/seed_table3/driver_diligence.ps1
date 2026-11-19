# Diligence driver for the Table III seed experiment (seed 22: industry, then fx29).
# Launched detached (WMI Win32_Process.Create) so it survives Claude app restarts.
# Logs: logs\driver_diligence.log (seed_experiment output), logs\diligence_cpu.log (% Processor Performance every 10 min)
$ErrorActionPreference = 'Continue'
$d = $PSScriptRoot   # (avoid a non-ASCII literal: PS 5.1 reads BOM-less .ps1 as ANSI)
Set-Location $d
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD = '1'
$env:PYTHONIOENCODING = 'utf-8'
$py = 'C:\Users\ando\CDSM\.venv\Scripts\python.exe'
$drvLog = Join-Path $d 'logs\driver_diligence.log'
$cpuLog = Join-Path $d 'logs\diligence_cpu.log'

function Write-DrvLog([string]$msg) {
    Add-Content -Path $drvLog -Value ("[{0}] [driver] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg) -Encoding UTF8
}

# CPU logger: separate hidden process that stops when this driver exits
$cpuProc = Start-Process powershell.exe -WindowStyle Hidden -PassThru -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$d\cpu_logger.ps1`" -WatchPid $PID -LogPath `"$cpuLog`""
Write-DrvLog "start (pid $PID, cpu logger pid $($cpuProc.Id))"

$common = '--repo C:\Users\ando\CDSM --machine diligence --data-root "%USERPROFILE%\.gluonts\datasets" --epoch-mc 3 --epoch-steps 300 --sample-repeats 3'
foreach ($ds in @('industry', 'fx29')) {
    Write-DrvLog "seed 22 / $ds : begin"
    cmd.exe /c "`"$py`" seed_experiment.py run --seeds 22 --datasets $ds $common >> `"$drvLog`" 2>&1"
    Write-DrvLog "seed 22 / $ds : exit $LASTEXITCODE"
}
Write-DrvLog 'all done'
