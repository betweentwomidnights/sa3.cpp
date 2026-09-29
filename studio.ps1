param(
    [int]$Port = 8006,
    [int]$TrainPort = 8016,
    [string]$Model = "medium",
    [string]$Encoding = "f16",
    [string]$BinDir = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($Port -lt 1 -or $Port -gt 65535 -or $TrainPort -lt 1 -or $TrainPort -gt 65535 -or $Port -eq $TrainPort) {
    throw "Port and TrainPort must be distinct valid TCP ports."
}

$candidates = if ($BinDir) { @($BinDir) } else { @(".", "build-cuda/bin/Release", "build-vulkan/bin/Release", "build-all/bin/Release", "build/bin/Release", "build-studio/bin/Release", "build-preview/bin/Release") }
$bin = $null
foreach ($candidate in $candidates) {
    $resolved = if ([IO.Path]::IsPathRooted($candidate)) { $candidate } else { Join-Path $root $candidate }
    if ((Test-Path (Join-Path $resolved "sa3-server.exe")) -and
        (Test-Path (Join-Path $resolved "sa3-train-web.exe")) -and
        (Test-Path (Join-Path $resolved "sa3-train.exe"))) {
        $bin = $resolved
        break
    }
}
if (-not $bin) { throw "Studio binaries not found. Run .\build.cmd cuda first, or pass -BinDir <path>." }

$trainUrl = "http://127.0.0.1:$TrainPort/api/health"
$ownsTrainer = $false
Push-Location $root
try {
    try { $health = Invoke-RestMethod -Uri $trainUrl -TimeoutSec 2 }
    catch { $health = $null }
    if ($health -and ($health.status -ne "ok" -or $health.studio_api -ne 2)) {
        if ($PSBoundParameters.ContainsKey("TrainPort")) {
            throw "Port $TrainPort has a different training server. Stop it or choose another -TrainPort."
        }
        $found = $false
        foreach ($candidate in 8017..8030) {
            try { $probe = Invoke-WebRequest -Uri "http://127.0.0.1:$candidate/api/health" -TimeoutSec 1; continue }
            catch {
                if ($_.Exception.Response) { continue }
                $TrainPort = $candidate
                $trainUrl = "http://127.0.0.1:$TrainPort/api/health"
                $health = $null
                $found = $true
                break
            }
        }
        if (-not $found) { throw "No free training port found from 8017 to 8030." }
    }
    if (-not $health) {
        $trainer = Start-Process -FilePath (Join-Path $bin "sa3-train-web.exe") -ArgumentList @("--host", "127.0.0.1", "--port", "$TrainPort") -WorkingDirectory $root -WindowStyle Hidden -PassThru
        $ownsTrainer = $true
        for ($i = 0; $i -lt 40; $i++) {
            if ($trainer.HasExited) { throw "The training server exited during startup." }
            try { $health = Invoke-RestMethod -Uri $trainUrl -TimeoutSec 1; if ($health.status -eq "ok") { break } }
            catch { }
            Start-Sleep -Milliseconds 250
        }
        if (-not $health -or $health.status -ne "ok") { throw "The training server did not become ready on port $TrainPort." }
    }

    $url = "http://127.0.0.1:$Port/"
    Write-Host "Studio: $url"
    Write-Host "Press Ctrl+C to stop."
    & (Join-Path $bin "sa3-server.exe") --model $Model --encoding $Encoding --port $Port --train-port $TrainPort
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    if ($ownsTrainer -and $trainer -and -not $trainer.HasExited) {
        Stop-Process -Id $trainer.Id -ErrorAction SilentlyContinue
    }
    Pop-Location
}
