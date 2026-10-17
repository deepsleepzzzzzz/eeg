$ErrorActionPreference = 'Continue'
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = Join-Path (Split-Path $PSScriptRoot -Parent) '.venv\Scripts\python.exe'
& $py src\tuning\scan_variants.py --variants .\configs\scan\fill_gear200_noise.yaml --task gear --condition 20_0 --workers 6 --seeds 42,43,44 --models cnn,gcn --device cuda --out-name variant_scan_fill *> reports\fill_gear200.log
Write-Output 'ALL_DONE'
