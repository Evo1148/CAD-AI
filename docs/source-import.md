# Clean source import

The canonical public import is based on the current **CAD AI V0.2** source archive, not the older M3-A snapshot.

The importer verifies current-architecture markers before copying anything: `DeterministicPromptGrounder`, `ResidualLLMExtractor`, `DeterministicPlateIntentGate`, Capability Pack 3, Prototype Usability Gate, and the LAB runner.

## Imported

- `src/cad_ai/`
- `tests/`
- `pyproject.toml`
- `uv.lock`
- source documentation under `docs/`

The original long-form Spanish development README is preserved as `docs/development-history.es.md`. The curated bilingual repository README remains authoritative for the public project page.

## Excluded

Generated or machine-local content is not part of the public source tree: `.venv/`, `artifacts/`, `build/`, `dist/`, `.pytest_cache/`, `__pycache__/`, `*.egg-info/`, generated STEP/STL outputs, and local model weights.

## Usage

From a fresh clone of this repository, temporarily allow the importer in the current PowerShell process, then run:

    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\\tools\\import-cad-ai-v02.ps1 -ZipPath "C:\\path\\to\\CAD_AI_V02.zip"

Then audit before committing:

    git status
    git add -N src tests pyproject.toml uv.lock docs
    git -c core.autocrlf=false --no-pager diff --stat
    git -c core.autocrlf=false --no-pager diff --name-only

Do not commit generated CAD outputs or local LLM model files.
