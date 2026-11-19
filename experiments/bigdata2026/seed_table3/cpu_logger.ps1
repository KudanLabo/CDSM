# Logs \Processor Information(_Total)\% Processor Performance every 10 minutes until process -WatchPid exits.
# ~17% means the EC throttling has come back.
param([int]$WatchPid, [string]$LogPath)
function Test-Alive { [bool](Get-Process -Id $WatchPid -ErrorAction SilentlyContinue) }
while (Test-Alive) {
    try {
        $v = [math]::Round((Get-Counter '\Processor Information(_Total)\% Processor Performance' -SampleInterval 5 -MaxSamples 1).CounterSamples[0].CookedValue, 1)
    } catch { $v = 'ERR' }
    Add-Content -Path $LogPath -Value ("{0}`t{1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $v) -Encoding UTF8
    for ($i = 0; $i -lt 60 -and (Test-Alive); $i++) { Start-Sleep -Seconds 10 }
}
Add-Content -Path $LogPath -Value ("{0}`tdriver {1} exited; logger stopped" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $WatchPid) -Encoding UTF8
