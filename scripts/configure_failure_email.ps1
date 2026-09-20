[CmdletBinding()]
param()

# The CLIXML SecureString is protected with Windows DPAPI for the current
# Windows user.  The resulting file is ignored by Git and cannot be reused on
# a different Windows account or computer.
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$secretDirectory = Join-Path $projectRoot "data\secrets"
$configPath = Join-Path $secretDirectory "gmail_alert.xml"

$email = Read-Host "Gmail address to send from"
if ([string]::IsNullOrWhiteSpace($email)) {
    throw "A Gmail address is required."
}
$recipient = Read-Host "Alert recipient (press Enter to use the same Gmail address)"
if ([string]::IsNullOrWhiteSpace($recipient)) {
    $recipient = $email
}
$appPassword = Read-Host "16-digit Gmail app password" -AsSecureString

New-Item -ItemType Directory -Force -Path $secretDirectory | Out-Null
[PSCustomObject]@{
    Email = $email
    Recipient = $recipient
    AppPassword = $appPassword
} | Export-Clixml -LiteralPath $configPath -Force

Write-Host "Encrypted failure-email settings saved outside Git: $configPath"
