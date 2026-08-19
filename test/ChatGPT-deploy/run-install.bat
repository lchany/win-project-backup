@echo off
setlocal
set DEST=C:\Temp\ChatGPT-deploy
set LOG=C:\Temp\chatgpt-install.log
mkdir "%DEST%" 2>nul
echo START %DATE% %TIME% > "%LOG%"
echo Copying files... >> "%LOG%"
copy /Y "\\tsclient\ChatGPT-deploy\ChatGPT-x64.msix" "%DEST%\" >> "%LOG%" 2>&1
copy /Y "\\tsclient\ChatGPT-deploy\ChatGPT-License.xml" "%DEST%\" >> "%LOG%" 2>&1
copy /Y "\\tsclient\ChatGPT-deploy\Install-ChatGPT.ps1" "%DEST%\" >> "%LOG%" 2>&1
echo Installing... >> "%LOG%"
powershell -ExecutionPolicy Bypass -NoProfile -File "%DEST%\Install-ChatGPT.ps1" >> "%LOG%" 2>&1
echo VERIFY >> "%LOG%"
powershell -NoProfile -Command "Get-AppxPackage | Where-Object { $_.Name -like 'OpenAI.*' } | Format-List" >> "%LOG%" 2>&1
echo DONE %DATE% %TIME% >> "%LOG%"
echo Installation finished. Log: %LOG%
timeout /t 5
