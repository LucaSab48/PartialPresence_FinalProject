$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

$generatorScript = Join-Path $projectRoot "generate_from_pico.py"
$displayScript = Join-Path $projectRoot "display_latest_image.py"
$audioDeviceName = "QuadCast"

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Write-Error "Python launcher 'py' was not found. Install Python for Windows and try again."
    exit 1
}

$generatorCommand = "Set-Location -LiteralPath '$projectRoot'; `$env:AUDIO_INPUT_DEVICE_NAME='$audioDeviceName'; py '$generatorScript'"
$displayCommand = "Set-Location -LiteralPath '$projectRoot'; py '$displayScript'"

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-Command", $generatorCommand
)

Start-Sleep -Milliseconds 500

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-Command", $displayCommand
)

Write-Host "Started generate_from_pico.py and display_latest_image.py in separate PowerShell windows."
Write-Host "USB microphone device hint for generator: $audioDeviceName"
