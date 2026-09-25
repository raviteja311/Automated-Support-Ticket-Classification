@echo off
REM Double-clickable wrapper for run.ps1.
REM PowerShell refuses to run .ps1 files on double-click by default, so this
REM invokes it explicitly with a bypass scoped to this one process only.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
pause
