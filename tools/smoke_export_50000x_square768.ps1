$ErrorActionPreference = 'Stop'
$src = 'D:\CNTDATA\coredata\selected_No28_No39_No41_No42\50000'
$dst = 'D:\CNTDATA\coredata\selected_No28_No39_No41_No42\50000_square768_smoke'
Remove-Item -Recurse -Force $dst -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $dst | Out-Null

$first = Get-ChildItem -Path $src -Filter '*.png' | Select-Object -First 1
if (-not $first) { throw 'No input images found' }

& 'D:\CNTDATA\CNTA_ML_Project\tools\export_50000x_square768.ps1' -InputDir $src -OutputDir $dst -Limit 1

$out = Join-Path $dst ($first.BaseName + '_square768.png')
if (-not (Test-Path -LiteralPath $out)) { throw 'Output not created' }
Write-Host 'OK'
