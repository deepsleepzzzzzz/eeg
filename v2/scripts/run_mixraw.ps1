$ErrorActionPreference = 'Continue'
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = '.\.venv\Scripts\python.exe'
foreach ($c in @(@{t='gear';c='20_0'},@{t='gear';c='30_2'})) {
  Write-Output ("[run ] mixraw {0} {1}  {2}" -f $c.t, $c.c, (Get-Date -Format 'HH:mm:ss'))
  & $py src\tuning\scan_variants.py --variants .\configs\scan\mixraw_v5.yaml --task $c.t --condition $c.c --workers 6 --seeds 42,43,44 --models cnn --device cuda --out-name variant_scan_mixraw
  Write-Output ("[done] mixraw {0} {1}  {2}" -f $c.t, $c.c, (Get-Date -Format 'HH:mm:ss'))
}
Write-Output 'ALL_DONE'
