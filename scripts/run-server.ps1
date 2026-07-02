param(
    [int]$Port = 8006,
    [string]$Model = "medium",
    [string]$Encoding = "f16",
    [string]$BuildDir = "build-cuda"
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ServerExe = Join-Path $RepoRoot (Join-Path $BuildDir "bin\Release\sa3-server.exe")

if (-not (Test-Path -LiteralPath $ServerExe)) {
    throw "sa3-server.exe not found at $ServerExe. Build first, for example: .\build.cmd cuda"
}

Set-Location $RepoRoot
Write-Host "[sa3-server] starting $ServerExe"
Write-Host "[sa3-server] http://127.0.0.1:$Port"
Write-Host "[sa3-server] close this terminal or press Ctrl+C to stop."
& $ServerExe --port $Port --model $Model --encoding $Encoding
