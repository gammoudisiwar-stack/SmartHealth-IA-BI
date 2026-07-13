<#
.SYNOPSIS
    Lance l'interface Streamlit du projet (inférence + cockpit BI).
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\run_app.ps1
#>
$ErrorActionPreference = "Stop"
$env:TF_CPP_MIN_LOG_LEVEL = "3"
$env:PYTHONIOENCODING = "utf-8"
Set-Location -Path $PSScriptRoot

$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) { throw "venv introuvable. Lancez d'abord ./run_all.ps1" }

Write-Host "Ouverture de l'interface Streamlit (http://localhost:8501) ..." -ForegroundColor Cyan
& $venvPy -m streamlit run chat_app.py
