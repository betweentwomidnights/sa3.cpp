@echo off
rem Start the inference and training web UIs together. See docs\SERVER.md.
powershell.exe -NoProfile -File "%~dp0studio.ps1" %*
