# One-command setup for Windows (PowerShell).
#   powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1          CPU stack
#   powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1 -Dl      + deep-learning stack
# Note: on Windows, PyPI's torch is CPU-only. For GPU work use the Linux compute machine.
param([switch]$Dl)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

# Run a native command and stop if it fails (PowerShell does not do this by itself).
function Run {
    & $args[0] $args[1..($args.Length - 1)]
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $($args -join ' ')" }
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git is missing. Install Git for Windows from https://git-scm.com/download/win and rerun."
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv (https://docs.astral.sh/uv/)..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:USERPROFILE\.cargo\bin;$env:Path"
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv installed but not on PATH yet. Open a new terminal and rerun this script."
    }
}

$py = (Get-Content .python-version -Raw).Trim()
Run uv python install $py

$syncArgs = @("sync", "--python", $py)
if ($Dl) { $syncArgs += @("--extra", "dl") }
Run uv @syncArgs

# strip notebook outputs on commit (per-clone git setting)
Run uv run nbstripout --install

Run uv run pytest -q
Write-Host ""
Write-Host "Setup done. Run everything through 'uv run', e.g. 'uv run pytest'."
