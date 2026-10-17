$ErrorActionPreference = 'Continue'
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = Join-Path (Split-Path $PSScriptRoot -Parent) '.venv\Scripts\python.exe'
foreach ($c in @(@{t='gear';c='20_0'},@{t='gear';c='30_2'})) {
  Write-Output ("[start] energy_stats {0} {1}  {2}" -f $c.t, $c.c, (Get-Date -Format 'HH:mm:ss'))
  & $py src\tuning\scan_variants.py --variants .\configs\scan\energy_stats_v5.yaml --task $c.t --condition $c.c --workers 6 --seeds 42,43,44 --models cnn --device cuda --out-name variant_scan_energy *>> reports\energy_stats.log
}
Write-Output 'ALL_DONE'
