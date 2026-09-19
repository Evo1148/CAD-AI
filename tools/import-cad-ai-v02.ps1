param(
    [Parameter(Mandatory = $true)]
    [string]$ZipPath
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ZipPath = (Resolve-Path $ZipPath).Path
$TempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("cad-ai-public-import-" + [guid]::NewGuid().ToString("N"))

function Assert-Exists {
    param([string]$Path, [string]$Label)
    if (-not (Test-Path $Path)) { throw "$Label not found: $Path" }
}

try {
    New-Item -ItemType Directory -Force $TempRoot | Out-Null
    Expand-Archive -LiteralPath $ZipPath -DestinationPath $TempRoot -Force
    $SourceRoot = $TempRoot

    Assert-Exists (Join-Path $SourceRoot "src\\cad_ai\\prompt_grounding.py") "Current CAD AI prompt grounding source"
    Assert-Exists (Join-Path $SourceRoot "src\\cad_ai\\capability_v02.py") "CAD AI V0.2 capability source"
    Assert-Exists (Join-Path $SourceRoot "src\\cad_ai\\prototype_gate.py") "Prototype usability gate"
    Assert-Exists (Join-Path $SourceRoot "src\\cad_ai\\lab.py") "LAB runner"
    Assert-Exists (Join-Path $SourceRoot "tests\\test_capability_cp3.py") "Capability Pack 3 tests"
    Assert-Exists (Join-Path $SourceRoot "pyproject.toml") "Python project metadata"

    $markerText = @(
        Get-Content (Join-Path $SourceRoot "src\\cad_ai\\prompt_grounding.py") -Raw
        Get-Content (Join-Path $SourceRoot "src\\cad_ai\\capability_v02.py") -Raw
        Get-Content (Join-Path $SourceRoot "README.md") -Raw
    ) -join "`n"

    foreach ($marker in @("DeterministicPromptGrounder", "DeterministicPlateIntentGate", "ResidualLLMExtractor", "Capability Pack 3", "Prototype Usability Gate")) {
        if (-not $markerText.Contains($marker)) { throw "Expected current-architecture marker not found: $marker" }
    }

    foreach ($dest in @("src", "tests")) {
        if (Test-Path (Join-Path $RepoRoot $dest)) { throw "$dest/ already exists in the repository. Import is intentionally one-shot." }
    }

    Copy-Item (Join-Path $SourceRoot "src") (Join-Path $RepoRoot "src") -Recurse
    Copy-Item (Join-Path $SourceRoot "tests") (Join-Path $RepoRoot "tests") -Recurse
    Copy-Item (Join-Path $SourceRoot "pyproject.toml") (Join-Path $RepoRoot "pyproject.toml")
    Copy-Item (Join-Path $SourceRoot "uv.lock") (Join-Path $RepoRoot "uv.lock")

    $SourceDocs = Join-Path $SourceRoot "docs"
    $RepoDocs = Join-Path $RepoRoot "docs"
    New-Item -ItemType Directory -Force $RepoDocs | Out-Null
    if (Test-Path $SourceDocs) {
        Get-ChildItem $SourceDocs -File | ForEach-Object { Copy-Item $_.FullName (Join-Path $RepoDocs $_.Name) }
    }
    if (Test-Path (Join-Path $SourceRoot "README.md")) {
        Copy-Item (Join-Path $SourceRoot "README.md") (Join-Path $RepoDocs "development-history.es.md")
    }

    Get-ChildItem (Join-Path $RepoRoot "src") -Directory -Recurse -Force |
        Where-Object { $_.Name -like "*.egg-info" -or $_.Name -eq "__pycache__" } |
        Sort-Object FullName -Descending | Remove-Item -Recurse -Force

    foreach ($name in @(".venv", "artifacts", "build", "dist", ".pytest_cache", "__pycache__")) {
        Get-ChildItem $RepoRoot -Directory -Recurse -Force -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -eq $name } |
            Sort-Object FullName -Descending | Remove-Item -Recurse -Force
    }

    $textExtensions = @(".py", ".md", ".toml", ".txt", ".json", ".yaml", ".yml")
    $files = Get-ChildItem (Join-Path $RepoRoot "src"), (Join-Path $RepoRoot "tests"), (Join-Path $RepoRoot "docs") -File -Recurse |
        Where-Object { $textExtensions -contains $_.Extension.ToLowerInvariant() }
    $patterns = @("github_pat_", "ghp_", "BEGIN OPENSSH PRIVATE KEY", "BEGIN RSA PRIVATE KEY", "C:\\Users\\", "/home/")
    $suspicious = $files | Select-String -Pattern $patterns -SimpleMatch
    if ($suspicious) {
        $suspicious | ForEach-Object { Write-Host $_ -ForegroundColor Red }
        throw "Safety scan found credential-like content or machine-specific absolute paths. Review before committing."
    }

    Write-Host ""
    Write-Host "CAD AI V0.2 import complete." -ForegroundColor Green
    Write-Host "Imported: src/, tests/, pyproject.toml, uv.lock and project documentation."
    Write-Host "Excluded: *.egg-info, virtual environments, artifacts, build outputs and caches."
    Write-Host ""
    Write-Host "Next:"
    Write-Host "  git status"
    Write-Host "  git add -N src tests pyproject.toml uv.lock docs"
    Write-Host "  git -c core.autocrlf=false --no-pager diff --stat"
}
finally {
    if (Test-Path $TempRoot) { Remove-Item $TempRoot -Recurse -Force -ErrorAction SilentlyContinue }
}
