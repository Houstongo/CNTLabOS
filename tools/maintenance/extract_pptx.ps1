$ppt_path = "d:\CNTDATA\CNTA_ML_Project\scripts\temp_zone.pptx"
$out_path = "d:\CNTDATA\CNTA_ML_Project\scripts\pptx_content.txt"

try {
    $ppt = New-Object -ComObject PowerPoint.Application
    $pres = $ppt.Presentations.Open($ppt_path, 0, 0, 0)
    $content = ""
    foreach ($slide in $pres.Slides) {
        $content += "Slide $($slide.SlideNumber):`n"
        foreach ($shape in $slide.Shapes) {
            if ($shape.HasTextFrame) {
                $content += $shape.TextFrame.TextRange.Text + "`n"
            }
            if ($shape.HasTable) {
                for ($r = 1; $r -le $shape.Table.Rows.Count; $r++) {
                    for ($c = 1; $c -le $shape.Table.Columns.Count; $c++) {
                        $content += $shape.Table.Cell($r, $c).Shape.TextFrame.TextRange.Text + "`t"
                    }
                    $content += "`n"
                }
            }
        }
        $content += "--------------------`n"
    }
    $content | Out-File -FilePath $out_path -Encoding utf8
    $pres.Close()
    $ppt.Quit()
    Write-Host "Success: Content saved to $out_path"
} catch {
    Write-Host "Error: $($_.Exception.Message)"
}
