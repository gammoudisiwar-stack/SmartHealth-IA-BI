<#
.SYNOPSIS
    Exécute de bout en bout le projet AI-BI de détection d'anémie (transfer learning + exports Power BI).

.DESCRIPTION
    1. Crée/active un environnement virtuel .venv
    2. Installe les dépendances (requirements.txt)
    3. Télécharge les datasets Mendeley si absents (via Invoke-WebRequest -> gère le proxy SSL d'entreprise)
    4. Enchaîne le pipeline : notebooks (préparation + entraînement/évaluation) -> bootstrap agent (PostgreSQL) -> build_pbip
    5. Affiche le récapitulatif des KPI

.PARAMETER SkipInstall
    Ne pas (ré)installer les dépendances.

.PARAMETER SkipDownload
    Ne pas télécharger les datasets (supposés déjà présents).

.PARAMETER Force
    Forcer le re-téléchargement des datasets.

.EXAMPLE
    ./run_all.ps1
    ./run_all.ps1 -SkipInstall -SkipDownload
#>
[CmdletBinding()]
param(
    [switch]$SkipInstall,
    [switch]$SkipDownload,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$ProgressPreference = "SilentlyContinue"
$env:TF_CPP_MIN_LOG_LEVEL = "3"      # réduit les logs TensorFlow
$env:PYTHONIOENCODING = "utf-8"      # évite les crashs d'encodage console (cp1252)

Set-Location -Path $PSScriptRoot

function Write-Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

# --------------------------------------------------------------------------- #
# 1. Python système
# --------------------------------------------------------------------------- #
Write-Step "Vérification de Python"
$sysPy = Get-Command python -ErrorAction SilentlyContinue
if (-not $sysPy) { $sysPy = Get-Command py -ErrorAction SilentlyContinue }
if (-not $sysPy) { throw "Python introuvable. Installez Python 3.11 puis relancez." }
Write-Host "  Python : $($sysPy.Source)"

# --------------------------------------------------------------------------- #
# 2. Environnement virtuel
# --------------------------------------------------------------------------- #
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Step "Création de l'environnement virtuel (.venv)"
    & $sysPy.Source -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Échec de la création du venv." }
}
Write-Host "  venv : $venvPy"

# --------------------------------------------------------------------------- #
# 3. Dépendances
# --------------------------------------------------------------------------- #
if (-not $SkipInstall) {
    Write-Step "Installation des dépendances (requirements.txt)"
    & $venvPy -m pip install --upgrade pip --disable-pip-version-check
    & $venvPy -m pip install -r requirements.txt --disable-pip-version-check
    if ($LASTEXITCODE -ne 0) { throw "Échec de l'installation des dépendances." }
} else {
    Write-Host "`n(Installation ignorée : -SkipInstall)"
}

# --------------------------------------------------------------------------- #
# 4. Datasets (téléchargement si absents)
# --------------------------------------------------------------------------- #
function Get-IfMissing($url, $outFile) {
    if ((Test-Path $outFile) -and -not $Force) {
        Write-Host "  (déjà présent) $outFile"
        return
    }
    Write-Host "  Téléchargement -> $outFile"
    Invoke-WebRequest -Uri $url -OutFile $outFile -UseBasicParsing
}

if (-not $SkipDownload) {
    Write-Step "Datasets Mendeley"
    New-Item -ItemType Directory -Force -Path "data" | Out-Null

    # Cohorte clinique (fine-tuning) — DOI 10.17632/tnswkb4bt8
    $clinUrl = "https://data.mendeley.com/public-files/datasets/tnswkb4bt8/files/714f5e03-3f87-462e-a065-09a25995fb6b/file_downloaded"
    Get-IfMissing $clinUrl "data\anemia_raw.xlsx"

    # Cohorte pédiatrique (pré-entraînement) — DOI 10.17632/y7v7ff3wpj
    $pedXlsx = Get-ChildItem -Path "data\pretrain_pediatric_x" -Recurse -Filter *.xlsx -ErrorAction SilentlyContinue | Select-Object -First 1
    if ((-not $pedXlsx) -or $Force) {
        Get-IfMissing "https://data.mendeley.com/public-api/zip/y7v7ff3wpj/download/1" "data\pretrain_pediatric.zip"
        Expand-Archive -Path "data\pretrain_pediatric.zip" -DestinationPath "data\pretrain_pediatric_x" -Force
        Write-Host "  Archive pédiatrique extraite."
    } else {
        Write-Host "  (déjà présent) dataset pédiatrique"
    }
} else {
    Write-Host "`n(Téléchargement ignoré : -SkipDownload)"
}

# --------------------------------------------------------------------------- #
# 5. Pipeline
# --------------------------------------------------------------------------- #
function Invoke-Step($title, $script, [string[]]$scriptArgs = @()) {
    Write-Step $title
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & $venvPy $script @scriptArgs
    if ($LASTEXITCODE -ne 0) { throw "Échec : $script (code $LASTEXITCODE)" }
    $sw.Stop()
    Write-Host ("  -> terminé en {0:N1}s" -f $sw.Elapsed.TotalSeconds) -ForegroundColor Green
}

$global:pipelineStart = [Diagnostics.Stopwatch]::StartNew()

function Invoke-Notebook($title, $nb) {
    Write-Step $title
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & $venvPy -m jupyter nbconvert --to notebook --execute --inplace $nb
    if ($LASTEXITCODE -ne 0) { throw "Échec du notebook : $nb" }
    $sw.Stop()
    Write-Host ("  -> terminé en {0:N1}s" -f $sw.Elapsed.TotalSeconds) -ForegroundColor Green
}

Invoke-Notebook "1/4 Préparation des données (notebook)"       "notebooks\1_data_preparation.ipynb"
Invoke-Notebook "2/4 Entraînement + évaluation (notebook)"     "notebooks\2_training_evaluation.ipynb"
Invoke-Step     "3/4 Peuplement base (agent -> PostgreSQL)"    "src\care_agent.py" @("--bootstrap")
Invoke-Step     "4/4 Tableau de bord Power BI (PostgreSQL)"    "src\build_pbip.py"
$global:pipelineStart.Stop()

# --------------------------------------------------------------------------- #
# 6. Récapitulatif
# --------------------------------------------------------------------------- #
Write-Step "TERMINÉ"
Write-Host ("Pipeline complet en {0:N1}s`n" -f $global:pipelineStart.Elapsed.TotalSeconds) -ForegroundColor Green

Write-Host "KPI métier (PostgreSQL / vue vw_ops_kpis) :" -ForegroundColor Cyan
& $venvPy -c "import sys; sys.path.append('src'); import procedures as p; [print(f'  {m:<32} {v}') for m, v in p.sp_get_kpis().items()]"
Write-Host "`nTableau de bord : ouvrir powerbi\AnemieCockpit.pbip (connecteur Npgsql requis)." -ForegroundColor Yellow
