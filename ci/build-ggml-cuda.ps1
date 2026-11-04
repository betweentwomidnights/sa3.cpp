# ABOUTME: Builds the portable ggml-cuda.dll on its own, with the CUDA runtime it needs.
# ABOUTME: Its output depends only on the ggml pin and toolchain, so runs can reuse it.
#
# ggml-cuda is over 90% of a Windows package build: 145 files, each compiled for
# every architecture in ggml's portable default. It is configured from ggml/
# directly rather than through sa3.cpp's CMakeLists.txt, so nothing outside the
# pin can change it, and -PrintKey names everything that can. release.yml keeps
# the result as an artifact under that key and package-windows.ps1 -PrebuiltCuda
# packages it.
#
# Usage:
#   ci\build-ggml-cuda.ps1 -PrintKey
#   ci\build-ggml-cuda.ps1 [-BuildDir build-cuda] [-OutDir ggml-cuda-bundle] [-Jobs 4]
param(
    [switch]$PrintKey,
    [string]$BuildDir = "build-cuda",
    [string]$OutDir = "ggml-cuda-bundle",
    [int]$Jobs = 4
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Fail([string]$message) {
    Write-Error "build-ggml-cuda: $message"
    exit 1
}

function Invoke-Checked([string]$program, [string[]]$arguments) {
    & $program @arguments
    if ($LASTEXITCODE -ne 0) { Fail "$program exited with $LASTEXITCODE" }
}

$vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { Fail "vswhere was not found" }
$instance = & $vswhere -version "[17.0,18.0)" -latest -property installationPath
if (-not $instance) { Fail "Visual Studio 2022 was not found" }
$toolsetFile = Join-Path $instance "VC\Auxiliary\Build\Microsoft.VCToolsVersion.default.txt"
if (-not (Test-Path $toolsetFile)) { Fail "$toolsetFile is missing" }
$toolset = (Get-Content $toolsetFile -Raw).Trim()

$ggmlCommit = (git -C ggml rev-parse HEAD).Trim()
if (-not $env:SA3_CUDA_VERSION) { Fail "SA3_CUDA_VERSION is not set (the full toolkit version, like 12.8.1)" }

if ($PrintKey) {
    # This script holds the configure flags, so a change to it is a change to the output.
    $script = (Get-FileHash -Algorithm SHA256 $PSCommandPath).Hash.Substring(0, 12).ToLowerInvariant()
    Write-Output "$($ggmlCommit.Substring(0, 12))-cuda$env:SA3_CUDA_VERSION-msvc$toolset-$script"
    exit 0
}

if (git -C ggml status --porcelain --untracked-files=no) { Fail "ggml has local changes; the bundle would not match its pin" }
if (-not $env:CUDA_PATH) { Fail "CUDA_PATH is not set; install the CUDA Toolkit" }
if ($Jobs -lt 1) { Fail "-Jobs must be positive" }

$cmake = (Get-Command cmake.exe -ErrorAction SilentlyContinue)
$cmake = if ($cmake) { $cmake.Source } else { Join-Path $instance "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" }
if (-not (Test-Path $cmake)) { Fail "CMake was not found on PATH or in Visual Studio 2022" }
Invoke-Checked $cmake @(
    "-S", "ggml", "-B", $BuildDir,
    "-G", "Visual Studio 17 2022", "-A", "x64",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_SHARED_LIBS=ON",
    "-DGGML_NATIVE=OFF",
    "-DGGML_BACKEND_DL=ON",
    "-DGGML_CPU_ALL_VARIANTS=ON",
    "-DGGML_METAL=OFF",
    "-DGGML_CUDA=ON",
    "-DGGML_BUILD_TESTS=OFF",
    "-DGGML_BUILD_EXAMPLES=OFF"
)
Invoke-Checked $cmake @("--build", $BuildDir, "--config", "Release", "--target", "ggml-cuda", "--parallel", "$Jobs")

$cudaMajor = ($env:SA3_CUDA_VERSION -split "\.")[0]
if (Test-Path $OutDir) { Remove-Item -Recurse -Force $OutDir }
New-Item -ItemType Directory -Force $OutDir | Out-Null
Copy-Item (Join-Path $BuildDir "bin\Release\ggml-cuda.dll") $OutDir
foreach ($name in "cudart64_$cudaMajor.dll", "cublas64_$cudaMajor.dll", "cublasLt64_$cudaMajor.dll") {
    Copy-Item (Join-Path $env:CUDA_PATH "bin\$name") $OutDir
}
Copy-Item (Join-Path $env:CUDA_PATH "EULA.txt") (Join-Path $OutDir "NVIDIA-CUDA-EULA.txt")

$info = [ordered]@{
    ggml_commit  = $ggmlCommit
    cuda         = $env:SA3_CUDA_VERSION
    msvc_toolset = $toolset
    source_commit = (git rev-parse HEAD).Trim()
    run_id       = $env:GITHUB_RUN_ID
}
[System.IO.File]::WriteAllText(
    (Join-Path $OutDir "BUNDLE-INFO.json"),
    ($info | ConvertTo-Json) + "`n",
    (New-Object System.Text.UTF8Encoding($false)))
Get-ChildItem $OutDir | ForEach-Object { Write-Host ("{0,-28} {1,8:N1} MB" -f $_.Name, ($_.Length / 1MB)) }
