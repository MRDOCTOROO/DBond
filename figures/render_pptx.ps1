param(
  [Parameter(Mandatory=$true)][string]$Pptx,
  [Parameter(Mandatory=$true)][string]$OutPng,
  [int]$Width = 2560,
  [int]$Height = 1440
)
$ErrorActionPreference = "Stop"
$abs = (Resolve-Path $Pptx).Path
$ppt = New-Object -ComObject PowerPoint.Application
try {
  $pres = $ppt.Presentations.Open($abs, $true, $false, $false)  # ReadOnly, no title
  $pres.Slides[1].Export($OutPng, "PNG", $Width, $Height)
  $pres.Close()
  Write-Output "OK $OutPng"
} finally {
  $ppt.Quit()
  [System.Runtime.Interopservices.Marshal]::ReleaseComObject($ppt) | Out-Null
}
