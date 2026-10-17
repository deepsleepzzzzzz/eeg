$ErrorActionPreference = 'Continue'
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = Join-Path (Split-Path $PSScriptRoot -Parent) '.venv\Scripts\python.exe'
Write-Output ("[start] lag_spacing {0}" -f (Get-Date -Format 'HH:mm:ss'))
& $py src\tuning\scan_variants.py --variants .\configs\scan\lag_spacing_v5.yaml --task gear --condition 30_2 --workers 6 --seeds 42,43,44 --models cnn,gcn --device cuda --out-name variant_scan_lagspace *> reports\lagspace.log
Write-Output ("[done ] lag_spacing {0}" -f (Get-Date -Format 'HH:mm:ss'))
Write-Output 'ALL_DONE'
