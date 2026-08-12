[CmdletBinding()]
param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$PythonInstallDir = (Join-Path $env:LOCALAPPDATA "QA-Agent-Python31210"),
    [switch]$SkipHashCheck
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$bundleRoot = $PSScriptRoot
$pythonInstaller = Join-Path $bundleRoot "python\python-3.12.10-amd64.exe"
$wheelDirectory = Join-Path $bundleRoot "wheels"
$requirementsFile = Join-Path $bundleRoot "requirements-offline.txt"
$venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

foreach ($requiredPath in @($pythonInstaller, $wheelDirectory, $requirementsFile)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Offline bundle is incomplete: $requiredPath"
    }
}

if (-not $SkipHashCheck) {
    & (Join-Path $bundleRoot "verify_offline.ps1") -BundleRoot $bundleRoot -SkipImportCheck
}

if (-not (Test-Path -LiteralPath $PythonInstallDir)) {
    $installerArguments = "/quiet InstallAllUsers=0 TargetDir=`"$PythonInstallDir`" PrependPath=0 Include_pip=1 Include_test=0 Include_launcher=0 Shortcuts=0"
    $installerProcess = Start-Process -FilePath $pythonInstaller -ArgumentList $installerArguments -PassThru -Wait
    if ($installerProcess.ExitCode -ne 0) {
        throw "Bundled Python installer failed with exit code $($installerProcess.ExitCode)."
    }
}

$basePython = Join-Path $PythonInstallDir "python.exe"
if (-not (Test-Path -LiteralPath $basePython)) {
    throw "Python was not installed at the expected path: $basePython"
}

if (Test-Path -LiteralPath $venvPython) {
    throw "Refusing to overwrite the existing virtual environment: $venvPython. Remove it manually only when you intend to recreate it."
}

& $basePython -m venv (Join-Path $ProjectRoot ".venv")
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create project virtual environment."
}

& $venvPython -m pip install --no-index --find-links $wheelDirectory --requirement $requirementsFile
if ($LASTEXITCODE -ne 0) {
    throw "Offline dependency installation failed."
}

& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) {
    throw "Installed environment failed pip check."
}

Write-Host "Offline environment created successfully: $venvPython"
