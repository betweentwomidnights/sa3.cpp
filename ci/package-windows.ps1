# ABOUTME: Builds the portable Windows packages a supervisor such as gary4local
# ABOUTME: installs: a core zip, one zip per GPU backend, the CUDA runtime, SHA256SUMS.
#
# The backends are built as dynamic libraries (GGML_BACKEND_DL) and the CPU
# backend in every instruction-set variant (GGML_CPU_ALL_VARIANTS), so one core
# package runs on any x64 machine and a GPU backend is a single DLL unpacked
# beside it. ggml looks for backend DLLs next to the executable, which is what
# makes the split work: unpack core, unpack one backend over it, run.
#
# GGML_NATIVE is off so nothing is tuned to the machine that built it, and no
# CUDA architecture list is passed: with native off, ggml's own default covers
# Maxwell through Blackwell as PTX plus real code for the common cards. The
# CUDA runtime is its own zip because it is most of the download and does not
# change between sa3.cpp releases; a supervisor installs it once and puts it on
# PATH, where ggml-cuda.dll's imports resolve from.
#
# The same script runs in .github/workflows/release.yml and on a developer
# machine, so a package built by hand is the package CI would have built.
#
# Usage:
#   ci\package-windows.ps1 -Version v0.1.0 [-BuildDir build-dist] [-OutDir dist] [-SkipTests]
#   ci\package-windows.ps1 -Version v0.1.0 -CpuOnly   # local packaging smoke check
#   ci\package-windows.ps1 -Version v0.1.0 -CudaArch native  # local GPU smoke check
param(
    [Parameter(Mandatory = $true)][string]$Version,
    [string]$BuildDir = "build-dist",
    [string]$OutDir = "dist",
    [switch]$SkipTests,
    [switch]$CpuOnly,
    [string]$CudaArch = "",
    [int]$Jobs = 4
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# The script clears its staging directory before each run. Keep that deletion
# strictly inside this checkout even if a caller passes an absolute BuildDir.
$rootPrefix = [System.IO.Path]::GetFullPath($root).TrimEnd('\') + '\'
$buildInput = if ([System.IO.Path]::IsPathRooted($BuildDir)) { $BuildDir } else { Join-Path $root $BuildDir }
$outInput = if ([System.IO.Path]::IsPathRooted($OutDir)) { $OutDir } else { Join-Path $root $OutDir }
$buildPath = [System.IO.Path]::GetFullPath($buildInput)
$outPath = [System.IO.Path]::GetFullPath($outInput)
if (-not $buildPath.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "BuildDir must be inside $root"
}
if (-not $outPath.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "OutDir must be inside $root"
}

function Fail([string]$message) {
    Write-Error "package-windows: $message"
    exit 1
}

function Invoke-Checked([string]$program, [string[]]$arguments) {
    & $program @arguments
    if ($LASTEXITCODE -ne 0) {
        Fail "$program exited with $LASTEXITCODE"
    }
}

# --- toolchain --------------------------------------------------------------

$cmake = (Get-Command cmake.exe -ErrorAction SilentlyContinue)
if ($cmake) {
    $cmake = $cmake.Source
    $ctest = Join-Path (Split-Path -Parent $cmake) "ctest.exe"
} else {
    foreach ($edition in "Community", "Professional", "Enterprise", "BuildTools") {
        $candidate = "C:\Program Files\Microsoft Visual Studio\2022\$edition\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
        if (Test-Path $candidate) {
            $cmake = $candidate
            $ctest = Join-Path (Split-Path -Parent $candidate) "ctest.exe"
            break
        }
    }
}
if (-not $cmake) { Fail "CMake was not found on PATH or in Visual Studio 2022" }
if (-not $CpuOnly -and -not $env:CUDA_PATH) { Fail "CUDA_PATH is not set; install the CUDA Toolkit" }
if (-not $CpuOnly -and -not $env:VULKAN_SDK) { Fail "VULKAN_SDK is not set; install the Vulkan SDK" }
if ($CpuOnly -and $CudaArch) { Fail "-CudaArch cannot be used with -CpuOnly" }
if ($CudaArch -and $BuildDir -eq "build-dist") { Fail "-CudaArch needs a separate -BuildDir to keep the portable release cache clean" }
if ($Jobs -lt 1) { Fail "-Jobs must be positive" }

# version.json ships with a full toolkit install; a trimmed CI install may only
# have the versioned folder, which is named for the same release (v12.8).
if (-not $CpuOnly) {
    $versionJson = Join-Path $env:CUDA_PATH "version.json"
    if (Test-Path $versionJson) {
        $cudaVersion = (Get-Content $versionJson -Raw | ConvertFrom-Json).cuda.version
    } else {
        $cudaVersion = (Split-Path -Leaf $env:CUDA_PATH).TrimStart("v")
    }
    if ($cudaVersion -notmatch "^\d+\.\d+") { Fail "cannot tell the CUDA version from $env:CUDA_PATH" }
    $cudaMajorMinor = ($cudaVersion -split "\.")[0..1] -join "."
    $cudaMajor = ($cudaVersion -split "\.")[0]
}

Write-Host "sa3.cpp    $(git rev-parse --short HEAD)"
Write-Host "ggml       $(git -C ggml rev-parse HEAD)"
Write-Host "version    $Version"
if (-not $CpuOnly) {
    Write-Host "cuda       $cudaVersion ($env:CUDA_PATH)"
    Write-Host "vulkan     $env:VULKAN_SDK"
}

# --- build ------------------------------------------------------------------

$gpuBuild = if ($CpuOnly) { "OFF" } else { "ON" }
$configure = @(
    "-S", ".", "-B", $buildPath,
    "-G", "Visual Studio 17 2022", "-A", "x64",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DGGML_NATIVE=OFF",
    "-DGGML_BACKEND_DL=ON",
    "-DGGML_CPU_ALL_VARIANTS=ON",
    "-DGGML_METAL=OFF",
    "-DSA3_CUDA=$gpuBuild",
    "-DSA3_VULKAN=$gpuBuild",
    "-DSA3_BUILD_TOOLS=ON",
    "-DSA3_BUILD_SAT=ON",
    "-DBUILD_TESTING=ON"
)
if ($CudaArch) { $configure += "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch" }
Invoke-Checked $cmake $configure
Invoke-Checked $cmake @("--build", $buildPath, "--config", "Release", "--parallel", "$Jobs")

# The fixture-free tests run on the CPU backend, which here is loaded
# dynamically exactly as it will be on a user's machine.
if (-not $SkipTests) {
    Invoke-Checked $ctest @("--test-dir", $buildPath, "-C", "Release", "--output-on-failure")
}

# --- stage ------------------------------------------------------------------

$bin = Join-Path $buildPath "bin\Release"
$stage = Join-Path $buildPath "package"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force $outPath | Out-Null

function Stage([string]$name, [string[]]$patterns, [string]$from) {
    $dir = Join-Path $stage $name
    New-Item -ItemType Directory -Force $dir | Out-Null
    foreach ($pattern in $patterns) {
        $found = @(Get-ChildItem -Path $from -Filter $pattern -File)
        if ($found.Count -eq 0) { Fail "$name package: nothing matches $pattern in $from" }
        foreach ($file in $found) { Copy-Item $file.FullName $dir }
    }
    return $dir
}

$coreDir = Stage "core" @(
    "sa3-server.exe",
    "sa3-train-web.exe",
    "sa3-train.exe",
    "sa3-generate.exe",
    "sat-generate.exe",
    "sa3-lora-convert.exe",
    "sa3-smoke.exe",
    "sa3.dll",
    "ggml.dll",
    "ggml-base.dll",
    "ggml-cpu-*.dll"
) $bin
Copy-Item (Join-Path $root "LICENSE") $coreDir
Copy-Item (Join-Path $root "docs\RUNTIME_RELEASE.md") (Join-Path $coreDir "RUNTIME_README.md")
Copy-Item (Join-Path $root "docs\THIRD_PARTY_NOTICES.md") (Join-Path $coreDir "THIRD_PARTY_NOTICES.md")
Copy-Item (Join-Path $root "ggml\LICENSE") (Join-Path $coreDir "LICENSE-ggml.txt")
Copy-Item (Join-Path $root "vendor\cpp-httplib\LICENSE") (Join-Path $coreDir "LICENSE-cpp-httplib.txt")
Copy-Item (Join-Path $root "vendor\yyjson\LICENSE") (Join-Path $coreDir "LICENSE-yyjson.txt")

# A GPU backend that landed in the core zip would load on every machine, and
# one missing from its own zip would never load anywhere. Check both ways.
foreach ($backend in "cuda", "vulkan") {
    if (Test-Path (Join-Path $coreDir "ggml-$backend.dll")) { Fail "ggml-$backend.dll leaked into the core package" }
}
if (-not $CpuOnly) {
    $cudaDir = Stage "cuda" @("ggml-cuda.dll") $bin
    $vulkanDir = Stage "vulkan" @("ggml-vulkan.dll") $bin

    $cudartDir = Stage "cudart" @(
        "cudart64_$cudaMajor.dll",
        "cublas64_$cudaMajor.dll",
        "cublasLt64_$cudaMajor.dll"
    ) (Join-Path $env:CUDA_PATH "bin")
    Copy-Item (Join-Path $env:CUDA_PATH "EULA.txt") (Join-Path $cudartDir "NVIDIA-CUDA-EULA.txt")
}

# --- zip and checksum -------------------------------------------------------

Add-Type -AssemblyName System.IO.Compression.FileSystem

$archives = [ordered]@{
    "sa3-$Version-windows-x64-core.zip" = $coreDir
}
if (-not $CpuOnly) {
    $archives["sa3-$Version-windows-x64-cuda.zip"] = $cudaDir
    $archives["sa3-$Version-windows-x64-vulkan.zip"] = $vulkanDir
    $archives["cudart-$cudaMajorMinor-windows-x64.zip"] = $cudartDir
}

$sums = New-Object System.Text.StringBuilder
foreach ($entry in $archives.GetEnumerator()) {
    $zip = Join-Path $outPath $entry.Key
    if (Test-Path $zip) { Remove-Item -Force $zip }
    [System.IO.Compression.ZipFile]::CreateFromDirectory(
        (Resolve-Path $entry.Value).Path, $zip,
        [System.IO.Compression.CompressionLevel]::Optimal, $false)
    $hash = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLowerInvariant()
    [void]$sums.Append("$hash  $($entry.Key)`n")
    $megabytes = [math]::Round((Get-Item $zip).Length / 1MB, 1)
    Write-Host ("{0,-44} {1,8} MB  {2}" -f $entry.Key, $megabytes, $hash)
}

# LF endings and no BOM, so `sha256sum -c SHA256SUMS` works as-is.
[System.IO.File]::WriteAllText(
    (Join-Path $outPath "SHA256SUMS"),
    $sums.ToString(),
    (New-Object System.Text.UTF8Encoding($false)))

Write-Host "packages -> $outPath"
