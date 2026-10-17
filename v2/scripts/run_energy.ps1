$ErrorActionPreference='Continue'
Set-Location (Split-Path $PSScriptRoot -Parent)
& '.\.venv\Scripts\python.exe' src\analysis\energy_only.py --reference per_window  *> reports\energy_perwindow.log
& '.\.venv\Scripts\python.exe' src\analysis\energy_only.py --reference global      *> reports\energy_global.log
Write-Output 'ALL_DONE'
