[CmdletBinding()]
param(
    [string]$BundleRoot = $PSScriptRoot,
    [switch]$SkipImportCheck
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$checksumFile = Join-Path $BundleRoot "SHA256SUMS.txt"
if (-not (Test-Path -LiteralPath $checksumFile)) {
    throw "Checksum manifest is missing: $checksumFile"
}

$verifiedCount = 0
foreach ($line in Get-Content -LiteralPath $checksumFile -Encoding UTF8) {
    if ([string]::IsNullOrWhiteSpace($line)) {
        continue
    }
    if ($line -notmatch "^(?<hash>[A-Fa-f0-9]{64}) \*(?<relative>.+)$") {
        throw "Invalid checksum line: $line"
    }
    $filePath = Join-Path $BundleRoot $Matches.relative
    if (-not (Test-Path -LiteralPath $filePath -PathType Leaf)) {
        throw "Checksum target is missing: $($Matches.relative)"
    }
    $actualHash = (Get-FileHash -LiteralPath $filePath -Algorithm SHA256).Hash
    if ($actualHash -ne $Matches.hash.ToUpperInvariant()) {
        throw "Checksum mismatch: $($Matches.relative)"
    }
    $verifiedCount++
}

if (-not $SkipImportCheck) {
    $venvPython = Join-Path (Split-Path -Parent $BundleRoot) ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython) {
        & $venvPython -c "import fastapi, qdrant_client, FlagEmbedding, fitz, rapidocr_onnxruntime; print('Offline runtime imports verified')"
        if ($LASTEXITCODE -ne 0) {
            throw "Offline runtime import check failed."
        }
    }
}

Write-Host "Verified $verifiedCount bundle files."
