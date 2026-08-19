# Automate ChatGPT install over full RDP session
$ErrorActionPreference = 'Continue'
Add-Type -AssemblyName System.Windows.Forms
Add-Type @"
using System;
using System.Runtime.InteropServices;
using System.Text;
public class Native {
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
    [DllImport("user32.dll")] public static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);
    public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint pid);
    [DllImport("user32.dll", CharSet=CharSet.Auto)] public static extern int GetWindowText(IntPtr hWnd, StringBuilder lpString, int nMaxCount);
    public const int SW_RESTORE = 9;
}
"@

function Focus-Mstsc([int]$ProcessId, [int]$TimeoutSec = 60) {
    for ($i = 0; $i -lt $TimeoutSec; $i++) {
        $found = $null
        [Native+EnumWindowsProc]$cb = {
            param($hWnd, $lParam)
            $pidOut = 0
            [Native]::GetWindowThreadProcessId($hWnd, [ref]$pidOut) | Out-Null
            if ($pidOut -eq $ProcessId) {
                $sb = New-Object System.Text.StringBuilder 256
                [Native]::GetWindowText($hWnd, $sb, 256) | Out-Null
                $script:found = $hWnd
                return $false
            }
            return $true
        }
        [Native]::EnumWindows($cb, [IntPtr]::Zero) | Out-Null
        if ($found) {
            [Native]::ShowWindow($found, [Native]::SW_RESTORE) | Out-Null
            [Native]::SetForegroundWindow($found) | Out-Null
            return $true
        }
        Start-Sleep -Seconds 1
    }
    return $false
}

$runCmd = '\\tsclient\ChatGPT-deploy\run-install.bat'
[System.Windows.Forms.Clipboard]::SetText($runCmd)

$rdpFile = 'C:\project\win-project-backup\test\ChatGPT-deploy\deploy-full.rdp'
Write-Host 'Launching RDP with drive redirect...'
$mstsc = Start-Process mstsc.exe -ArgumentList "`"$rdpFile`"" -PassThru

Write-Host 'Waiting for RDP window...'
if (-not (Focus-Mstsc -ProcessId $mstsc.Id -TimeoutSec 90)) {
    Write-Host 'WARNING: Could not focus RDP window. Trying SendKeys anyway...'
}
Start-Sleep -Seconds 5

Write-Host "Sending install command: $runCmd"
[System.Windows.Forms.SendKeys]::SendWait('^{ESC}')
Start-Sleep -Milliseconds 800
[System.Windows.Forms.SendKeys]::SendWait('#{R}')
Start-Sleep -Seconds 2
[System.Windows.Forms.SendKeys]::SendWait('^v')
Start-Sleep -Milliseconds 500
[System.Windows.Forms.SendKeys]::SendWait('{ENTER}')

Write-Host 'Install started. Waiting up to 25 minutes...'
for ($i = 1; $i -le 150; $i++) {
    Start-Sleep -Seconds 10
    if (-not (Get-Process -Id $mstsc.Id -ErrorAction SilentlyContinue)) {
        Write-Host "RDP window closed after $($i*10)s"
        break
    }
    if ($i % 12 -eq 0) { Write-Host "Still running... ($($i*10)s elapsed)" }
}
Write-Host 'Install phase complete.'
