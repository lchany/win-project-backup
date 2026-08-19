# ChatGPT Desktop offline install script for Windows Server 2022
# Run as Administrator on the target server

$ErrorActionPreference = 'Stop'
$DeployDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$MsixPath = Join-Path $DeployDir 'ChatGPT-x64.msix'
$LicensePath = Join-Path $DeployDir 'ChatGPT-License.xml'

if (-not (Test-Path $MsixPath)) {
    throw "Missing package: $MsixPath"
}

Write-Host "Installing ChatGPT Desktop from $MsixPath ..."

if (Test-Path $LicensePath) {
    Add-AppxPackage -Path $MsixPath -LicensePath $LicensePath
} else {
    Add-AppxPackage -Path $MsixPath
}

$pkg = Get-AppxPackage -Name 'OpenAI.Codex' -ErrorAction SilentlyContinue
if (-not $pkg) {
    $pkg = Get-AppxPackage | Where-Object { $_.Name -like 'OpenAI.*' -or $_.InstallLocation -like '*ChatGPT*' } | Select-Object -First 1
}

if ($pkg) {
    Write-Host "Install succeeded."
    Write-Host "Package: $($pkg.Name) $($pkg.Version)"
    Write-Host "Location: $($pkg.InstallLocation)"
} else {
    throw 'Install finished but ChatGPT package was not found.'
}
