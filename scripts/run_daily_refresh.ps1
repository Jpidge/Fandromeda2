[CmdletBinding()]
param(
    [int]$DebugPort = 9222
)

# This runner intentionally works only in the signed-in desktop session.  The
# native Yahoo copy needs a visible Chrome window, and it fails without
# touching data/my_roster.txt if Yahoo has signed the user out.
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$logDirectory = Join-Path $projectRoot "data\logs"
$profileDirectory = Join-Path $projectRoot "data\yahoo_manual_chrome_profile"
$secretDirectory = Join-Path $projectRoot "data\secrets"
$alertConfigPath = Join-Path $secretDirectory "gmail_alert.xml"
$leagueUrl = "https://football.fantasysports.yahoo.com/f1/893771"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDirectory "daily_refresh_$timestamp.log"

New-Item -ItemType Directory -Force -Path $logDirectory, $profileDirectory | Out-Null
Start-Transcript -Path $logPath -Force | Out-Null

function Send-FandromedaFailureAlert {
    param([string]$FailureMessage)

    if (-not (Test-Path $alertConfigPath)) {
        Write-Warning "Failure email is not configured. See scripts\configure_failure_email.ps1."
        return
    }

    try {
        $alert = Import-Clixml -LiteralPath $alertConfigPath
        $credential = New-Object System.Management.Automation.PSCredential(
            $alert.Email,
            $alert.AppPassword
        )
        $message = New-Object System.Net.Mail.MailMessage
        $message.From = $alert.Email
        $message.To.Add($alert.Recipient)
        $message.Subject = "FANDROMEDA daily refresh failed"
        $message.Body = @"
FANDROMEDA's scheduled refresh did not complete.

Error: $FailureMessage
Log: $logPath
Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss K')
"@

        $smtp = New-Object System.Net.Mail.SmtpClient("smtp.gmail.com", 587)
        $smtp.EnableSsl = $true
        $smtp.Credentials = $credential.GetNetworkCredential()
        $smtp.Send($message)
        $message.Dispose()
        $smtp.Dispose()
        Write-Host "Sent failure alert to $($alert.Recipient)."
    }
    catch {
        Write-Warning "Could not send failure email: $($_.Exception.Message)"
    }
}

try {
    # Authorized by the project owner: prevent an existing Chrome session from
    # confusing the native Yahoo capture or CDP attachment.
    $existingChrome = Get-Process -Name chrome -ErrorAction SilentlyContinue
    if ($existingChrome) {
        Write-Host "Closing existing Chrome windows..."
        $existingChrome | Stop-Process -Force
        Start-Sleep -Seconds 2
    }

    $chromeCandidates = @(
        (Join-Path $env:ProgramFiles "Google\Chrome\Application\chrome.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Google\Chrome\Application\chrome.exe")
    ) | Where-Object { Test-Path $_ }
    if (-not $chromeCandidates) {
        throw "Google Chrome was not found in its standard installation location."
    }
    # A one-item PowerShell pipeline becomes a scalar string; wrap it before
    # indexing so Chrome's full path, not its first character, is launched.
    $chromeExe = @($chromeCandidates)[0]

    $python = (Get-Command python -ErrorAction Stop).Source
    $chromeArguments = @(
        "--remote-debugging-port=$DebugPort",
        "--user-data-dir=$profileDirectory",
        $leagueUrl
    )
    Write-Host "Opening the persistent Yahoo Chrome profile..."
    Start-Process -FilePath $chromeExe -ArgumentList $chromeArguments | Out-Null

    $debugUrl = "http://127.0.0.1:$DebugPort/json/version"
    $browserReady = $false
    for ($attempt = 1; $attempt -le 20; $attempt++) {
        try {
            Invoke-WebRequest -Uri $debugUrl -UseBasicParsing -TimeoutSec 2 | Out-Null
            $browserReady = $true
            break
        }
        catch {
            Start-Sleep -Seconds 1
        }
    }
    if (-not $browserReady) {
        throw "Chrome did not expose its local debugging connection on port $DebugPort."
    }

    Push-Location $projectRoot
    try {
        Write-Host "Refreshing Yahoo roster for FANDROMEDA's upcoming week..."
        & $python index.py `
            --refresh-yahoo-roster `
            --yahoo-attach-port $DebugPort `
            --yahoo-native-copy `
            --yahoo-week auto `
            --yahoo-no-confirm
        if ($LASTEXITCODE -ne 0) {
            throw "Yahoo roster refresh failed with exit code $LASTEXITCODE."
        }

        Write-Host "Rebuilding FANDROMEDA without opening a dashboard tab..."
        & $python index.py --no-open-dashboard
        if ($LASTEXITCODE -ne 0) {
            throw "Dashboard rebuild failed with exit code $LASTEXITCODE."
        }
    }
    finally {
        Pop-Location
    }

    Write-Host "Daily FANDROMEDA refresh completed successfully."
}
catch {
    $failureMessage = $_.Exception.Message
    Send-FandromedaFailureAlert -FailureMessage $failureMessage
    Write-Error "Daily FANDROMEDA refresh failed: $failureMessage" -ErrorAction Continue
    throw
}
finally {
    Stop-Transcript | Out-Null
}
