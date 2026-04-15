# Local pipeline runner
# Usage: .\run_local.ps1 "C:\path\to\input.csv"
#        .\run_local.ps1 "C:\path\to\input.csv" -Resume    # pick up where you left off
# Output will be saved next to the input file with _results and _results_passed suffixes

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$InputCsv,

    [int]$Limit = 0,

    [switch]$DryRun,

    [switch]$Resume
)

$ErrorActionPreference = "Stop"

# Resolve paths
$ProjectDir = "C:\Users\mitch\AffiliatePipeline"
$Python = "$ProjectDir\.venv\Scripts\python.exe"
$InputCsv = (Resolve-Path $InputCsv).Path

# Build output path next to input file
$InputDir = Split-Path $InputCsv
$InputName = [System.IO.Path]::GetFileNameWithoutExtension($InputCsv)
$OutputCsv = Join-Path $InputDir "${InputName}_results.csv"

# Build arguments
$Args = @("-m", "src.main", "--mode", "cli", "--input", $InputCsv, "--output", $OutputCsv, "-v")

if ($Limit -gt 0) {
    $Args += @("--limit", $Limit)
}

if ($DryRun) {
    $Args += "--dry-run"
}

if ($Resume) {
    $Args += "--resume"
}

Write-Host ""
Write-Host "=== Affiliate Pipeline - Local Run ===" -ForegroundColor Cyan
Write-Host "Input:  $InputCsv"
Write-Host "Output: $OutputCsv"
Write-Host ""

# Run from project directory so .env is picked up
Push-Location $ProjectDir
try {
    & $Python @Args
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "Done! Results saved to:" -ForegroundColor Green
Write-Host "  Full:   $OutputCsv"
Write-Host "  Passed: $($OutputCsv -replace '\.csv$', '_passed.csv')"
