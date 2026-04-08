param(
    [Parameter(Mandatory = $true)][string]$InputDir,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [int]$TargetSize = 768,
    [int]$Limit = 0
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing

if (-not (Test-Path -LiteralPath $InputDir)) {
    throw "InputDir not found: $InputDir"
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

function Save-ResizedSquare {
    param([string]$InputPath, [string]$OutputPath, [int]$Size)
    $src = [System.Drawing.Bitmap]::FromFile($InputPath)
    try {
        $scale = [Math]::Min($Size / [double]$src.Width, $Size / [double]$src.Height)
        $newW = [Math]::Max(1, [int][Math]::Round($src.Width * $scale))
        $newH = [Math]::Max(1, [int][Math]::Round($src.Height * $scale))
        $left = [int](($Size - $newW) / 2)
        $top = [int](($Size - $newH) / 2)

        $dst = New-Object System.Drawing.Bitmap($Size, $Size)
        try {
            $g = [System.Drawing.Graphics]::FromImage($dst)
            try {
                $g.Clear([System.Drawing.Color]::Black)
                $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
                $g.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
                $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
                $g.DrawImage($src, $left, $top, $newW, $newH)
            }
            finally { $g.Dispose() }
            $dst.Save($OutputPath, [System.Drawing.Imaging.ImageFormat]::Png)
        }
        finally { $dst.Dispose() }
    }
    finally { $src.Dispose() }
}

$files = Get-ChildItem -Path $InputDir -Filter '*.png' | Sort-Object Name
if ($Limit -gt 0) { $files = $files | Select-Object -First $Limit }

foreach ($file in $files) {
    $outName = $file.BaseName + '_square768.png'
    $outPath = Join-Path $OutputDir $outName
    if (Test-Path -LiteralPath $outPath) { continue }
    Save-ResizedSquare -InputPath $file.FullName -OutputPath $outPath -Size $TargetSize
}
