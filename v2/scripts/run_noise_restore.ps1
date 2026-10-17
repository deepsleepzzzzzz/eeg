$ErrorActionPreference = 'Continue'
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = Join-Path (Split-Path $PSScriptRoot -Parent) '.venv\Scripts\python.exe'
foreach ($c in @(@{t='gear';c='20_0'},@{t='bearing';c='20_0'},@{t='bearing';c='30_2'})) {
  Write-Output ("[start] noise_restore {0} {1}  {2}" -f $c.t, $c.c, (Get-Date -Format 'HH:mm:ss'))
  & $py src\tuning\scan_variants.py --variants .\configs\scan\noise_restore_v5.yaml --task $c.t --condition $c.c --workers 6 --seeds 42,43,44 --models cnn,gcn --device cuda --out-name variant_scan_noise_restore *>> reports\noise_restore.log
}
Write-Output 'ALL_DONE'
