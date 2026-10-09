param(
  [Parameter(Mandatory=$true)][string]$Pptx,
  [Parameter(Mandatory=$true)][string]$OutPdf
)
$ErrorActionPreference = "Stop"
$abs = (Resolve-Path $Pptx).Path
$out = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutPdf)
$ppt = New-Object -ComObject PowerPoint.Application
try {
  $pres = $ppt.Presentations.Open($abs, $true, $false, $false)  # ReadOnly, no window
  $pres.SaveAs($out, 32)  # ppSaveAsPDF
  $pres.Close()
  Write-Output "OK $out"
} finally {
  $ppt.Quit()
  [System.Runtime.Interopservices.Marshal]::ReleaseComObject($ppt) | Out-Null
}
