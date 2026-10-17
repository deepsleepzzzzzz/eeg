$ErrorActionPreference = 'Continue'
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = '.\.venv\Scripts\python.exe'
$conds = @(@{task='gear';cond='20_0'},@{task='gear';cond='30_2'},@{task='bearing';cond='20_0'},@{task='bearing';cond='30_2'})
foreach ($c in $conds) {
  Write-Output ("[run ] global-high task={0} {1}  {2}" -f $c.task, $c.cond, (Get-Date -Format 'HH:mm:ss'))
  & $py src\tuning\scan_variants.py --variants .\configs\scan\noise_global_high_v5.yaml --task $c.task --condition $c.cond --workers 8 --seeds 42,43,44 --device cuda
  Write-Output ("[done] global-high task={0} {1}  {2}" -f $c.task, $c.cond, (Get-Date -Format 'HH:mm:ss'))
}
Write-Output 'ALL_DONE'
