$ErrorActionPreference = 'Continue'
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = '.\.venv\Scripts\python.exe'
$conds = @(@{task='gear';cond='20_0'},@{task='gear';cond='30_2'},@{task='bearing';cond='20_0'},@{task='bearing';cond='30_2'})
foreach ($c in $conds) {
  Write-Output ("[run ] baseline task={0} {1}  {2}" -f $c.task, $c.cond, (Get-Date -Format 'HH:mm:ss'))
  & $py src\tuning\scan_variants.py --variants .\configs\scan\baseline_v5.yaml --task $c.task --condition $c.cond --workers 6 --seeds 42,43,44 --models raw_cnn,wdcnn,resnet1d --device cuda
  Write-Output ("[done] baseline task={0} {1}  {2}" -f $c.task, $c.cond, (Get-Date -Format 'HH:mm:ss'))
}
Write-Output 'ALL_DONE'
