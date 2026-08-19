# Deploy ChatGPT Desktop to a remote Windows host via WinRM (no RDP)
# Usage: .\Deploy-Remote.ps1 -ComputerName 103.236.93.62 -Credential (Get-Credential)

param(
    [Parameter(Mandatory = $true)]
    [string] $ComputerName,

    [Parameter(Mandatory = $true)]
    [pscredential] $Credential,

    [int] $Port = 5985,
    [switch] $UseSSL
)

$ErrorActionPreference = 'Stop'
$DeployDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RemoteDir = 'C:\Temp\ChatGPT-deploy'

$sessionParams = @{
    ComputerName = $ComputerName
    Credential   = $Credential
    Port         = $Port
}
if ($UseSSL) {
    $sessionParams.UseSSL = $true
    $sessionParams.SessionOption = New-PSSessionOption -SkipCACheck -SkipCNCheck
}

Write-Host "Creating WinRM session to $ComputerName`:$Port ..."
$session = New-PSSession @sessionParams

try {
    Invoke-Command -Session $session -ScriptBlock {
        param($RemoteDir)
        New-Item -ItemType Directory -Force -Path $RemoteDir | Out-Null
    } -ArgumentList $RemoteDir

    Write-Host 'Copying install package (this may take several minutes)...'
    Copy-Item -Path (Join-Path $DeployDir '*') -Destination $RemoteDir -ToSession $session -Force -Recurse

    Write-Host 'Running remote install...'
    Invoke-Command -Session $session -ScriptBlock {
        param($RemoteDir)
        Set-Location $RemoteDir
        & (Join-Path $RemoteDir 'Install-ChatGPT.ps1')
    } -ArgumentList $RemoteDir
}
finally {
    Remove-PSSession $session
}

Write-Host 'Remote deployment completed.'
