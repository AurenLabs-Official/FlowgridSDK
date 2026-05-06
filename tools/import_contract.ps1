param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$SourcePath,
    [Parameter(Mandatory = $true, Position = 1)]
    [string]$DestinationRelativeToRepoRoot
)

$ErrorActionPreference = "Stop"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$dest = Join-Path $repo $DestinationRelativeToRepoRoot

$raw = Get-Content -Raw -Path $SourcePath
# Drop obvious key-like substrings; always inspect the file before committing.
$redacted = $raw `
    -replace 'sk-ant-api\d\d-[a-zA-Z0-9_-]+', 'REDACTED' `
    -replace 'sk-[a-zA-Z0-9]{10,}', 'REDACTED' `
    -replace 'api-key:\s*[^\s]+', 'api-key: REDACTED'

# Validate JSON after redaction
$null = $redacted | ConvertFrom-Json

$dir = Split-Path -Parent $dest
if (-not (Test-Path $dir)) {
    New-Item -ItemType Directory -Path $dir | Out-Null
}
[System.IO.File]::WriteAllText($dest, $redacted.TrimEnd() + "`n", [System.Text.UTF8Encoding]::new($false))

Write-Host "Wrote $dest — review for secrets before git add."
