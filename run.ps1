<#
.SYNOPSIS
    Start the classifier and open the test page in your browser.

.DESCRIPTION
    One command to go from nothing to a working UI. Finds the project's virtual
    environment without needing it activated, starts the API, waits until it
    actually answers, then opens http://127.0.0.1:<port>/ui

    Press Ctrl+C in this window to stop the server.

.PARAMETER Port
    Port to serve on. Defaults to 8000.

.PARAMETER NoBrowser
    Start the server but do not open a browser.

.EXAMPLE
    .\run.ps1
    .\run.ps1 -Port 8010
    .\run.ps1 -NoBrowser
#>
param(
    [int]$Port = 8000,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Use the project's interpreter directly, so this works whether or not the
# environment is activated. MLOps is this repo's venv; .venv is what a fresh
# clone creates; falling back to PATH covers everything else.
$python = "python"
if (Test-Path ".\.venv\Scripts\python.exe") { $python = ".\.venv\Scripts\python.exe" }
if (Test-Path ".\MLOps\Scripts\python.exe") { $python = ".\MLOps\Scripts\python.exe" }

# A server already on this port would make the new one fail to bind, or worse,
# silently serve the old code. Say so rather than letting it confuse you.
$inUse = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($inUse) {
    Write-Host "Port $Port is already in use." -ForegroundColor Yellow
    Write-Host "Stop it first (docker compose down, or Ctrl+C the other server),"
    Write-Host "or pick another port:  .\run.ps1 -Port 8010"
    exit 1
}

$url = "http://127.0.0.1:$Port/ui"

if (-not $NoBrowser) {
    # Poll in a background job so the server keeps the foreground and Ctrl+C
    # still stops it cleanly. Opening too early shows a connection error page.
    Start-Job -ScriptBlock {
        param($OpenUrl, $HealthPort)
        for ($i = 0; $i -lt 60; $i++) {
            try {
                Invoke-WebRequest "http://127.0.0.1:$HealthPort/health" `
                    -TimeoutSec 2 -UseBasicParsing | Out-Null
                Start-Process $OpenUrl
                return
            } catch {
                Start-Sleep -Milliseconds 500
            }
        }
    } -ArgumentList $url, $Port | Out-Null
}

Write-Host ""
Write-Host "  Support Ticket Classifier" -ForegroundColor Cyan
Write-Host "  UI:    $url"
Write-Host "  Docs:  http://127.0.0.1:$Port/docs"
Write-Host "  Stop:  Ctrl+C"
Write-Host ""

& $python -m uvicorn automated_support_ticket_classification.api.app:app `
    --host 127.0.0.1 --port $Port
