# 补另外 3 个工况的噪声曲线（v5，9 档 SNR × 2 模型 × 3 seed）
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:CUDA_VISIBLE_DEVICES = ''
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
Remove-Item Env:SEU_DEVICE -ErrorAction SilentlyContinue
foreach ($c in @(@('gear','20_0'), @('bearing','30_2'), @('bearing','20_0'))) {
  $t = $c[0]; $k = $c[1]
  Write-Output ">>> $t/$k 开始"
  & '.\.venv\Scripts\python.exe' '.\src\tuning\scan_variants.py' --variants '.\configs\scan\noise_v5.yaml' --workers 8 --seeds 42,43,44 --device cuda --task $t --condition $k
  Write-Output ">>> $t/$k 结束 (exit=$LASTEXITCODE)"
}
Write-Output "ALL DONE"
