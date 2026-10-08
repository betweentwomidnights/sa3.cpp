# ABOUTME: Builds split Windows packages for gary4local and one standalone zip
# ABOUTME: with both GPU backends and the CUDA runtime, plus SHA256SUMS.
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
# Gary4local installs the shared CUDA runtime once from its own release. The
# standalone zip includes it so direct users need only one archive. -CudaRuntime
# also emits a separate runtime zip for local supervisor install testing.
#
# The same script runs in .github/workflows/release.yml and on a developer
# machine, so a package built by hand is the package CI would have built.
#
# Usage:
#   ci\package-windows.ps1 -Version v0.1.0 [-BuildDir build-dist] [-OutDir dist] [-SkipTests] [-CudaRuntime]
#   ci\package-windows.ps1 -Version v0.1.0 -CpuOnly   # local packaging smoke check
#   ci\package-windows.ps1 -Version v0.1.0 -CudaArch native  # local GPU smoke check
param(
    [Parameter(Mandatory = $true)][string]$Version,
    [string]$BuildDir = "build-dist",
    [string]$OutDir = "dist",
    [switch]$SkipTests,
    [switch]$CpuOnly,
    [switch]$CudaRuntime,
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
if ($Version -notmatch '^v\d+\.\d+\.\d+$') { throw "Version must look like v0.1.0" }

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
if ($CpuOnly -and $CudaRuntime) { Fail "-CudaRuntime requires a GPU build" }
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

$bin = Join-Path $buildPath "bin\Release"
foreach ($server in "sa3-server.exe", "sat-server.exe") {
    $reported = (& (Join-Path $bin $server) --version).Trim()
    if ("v$reported" -ne $Version) {
        Fail "$server reports $reported but the package is $Version; update project(VERSION) in CMakeLists.txt"
    }
}

# --- stage ------------------------------------------------------------------

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
    "sat-server.exe",
    "sa3-lora-convert.exe",
    "sa3-audio-analyze.exe",
    "sa3-smoke.exe",
    "sa3.dll",
    "ggml.dll",
    "ggml-base.dll",
    "ggml-cpu-*.dll"
) $bin
Copy-Item (Join-Path $root "LICENSE") $coreDir
Copy-Item (Join-Path $root "studio.cmd") $coreDir
Copy-Item (Join-Path $root "studio.ps1") $coreDir
Copy-Item (Join-Path $root "docs\RUNTIME_RELEASE.md") (Join-Path $coreDir "RUNTIME_README.md")
Copy-Item (Join-Path $root "docs\SAT_SERVER.md") (Join-Path $coreDir "SAT_SERVER.md")
Copy-Item (Join-Path $root "docs\AUDIO_ANALYSIS.md") (Join-Path $coreDir "AUDIO_ANALYSIS.md")
Copy-Item (Join-Path $root "docs\THIRD_PARTY_NOTICES.md") (Join-Path $coreDir "THIRD_PARTY_NOTICES.md")
Copy-Item (Join-Path $root "ggml\LICENSE") (Join-Path $coreDir "LICENSE-ggml.txt")
Copy-Item (Join-Path $root "vendor\cpp-httplib\LICENSE") (Join-Path $coreDir "LICENSE-cpp-httplib.txt")
Copy-Item (Join-Path $root "vendor\yyjson\LICENSE") (Join-Path $coreDir "LICENSE-yyjson.txt")
Copy-Item (Join-Path $root "vendor\signalsmith-stretch\LICENSE.txt") (Join-Path $coreDir "LICENSE-signalsmith-stretch.txt")
Copy-Item (Join-Path $root "vendor\signalsmith-linear\LICENSE.txt") (Join-Path $coreDir "LICENSE-signalsmith-linear.txt")

$buildInfo = [ordered]@{
    service     = "sa3"
    version     = $Version
    commit      = (git rev-parse HEAD).Trim()
    dirty       = [bool](git status --porcelain --untracked-files=no)
    ggml_commit = (git -C ggml rev-parse HEAD).Trim()
    platform    = "windows-x64"
    backends    = @()
    cuda        = $(if ($CpuOnly) { $null } else { $cudaVersion })
    vulkan_sdk  = $(if ($CpuOnly) { $null } else { Split-Path -Leaf $env:VULKAN_SDK })
    built_utc   = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
}
if (-not $CpuOnly) { $buildInfo.backends = @("cuda", "vulkan") }
[System.IO.File]::WriteAllText(
    (Join-Path $coreDir "BUILD-INFO.json"),
    ($buildInfo | ConvertTo-Json) + "`n",
    (New-Object System.Text.UTF8Encoding($false)))

# Verify the staged tools, not the build tree. Require Python even with
# -SkipTests so packaging cannot silently omit the host contract checks.
$pythonEntry = Get-Content (Join-Path $buildPath "CMakeCache.txt") |
    Where-Object { $_ -match '^_?Python3_EXECUTABLE:(INTERNAL|FILEPATH)=.+$' } |
    Select-Object -First 1
if (-not $pythonEntry) { Fail "Python 3 is required to verify the staged package" }
$python = ($pythonEntry -split '=', 2)[1]
Invoke-Checked $python @((Join-Path $root "ci/check-runtime-package.py"), $coreDir, $Version)

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

    $standaloneDir = Join-Path $stage "standalone"
    New-Item -ItemType Directory -Force $standaloneDir | Out-Null
    foreach ($part in $coreDir, $cudaDir, $vulkanDir, $cudartDir) {
        Get-ChildItem -Path $part -File | Copy-Item -Destination $standaloneDir
    }
    Copy-Item (Join-Path $root "models.cmd") $standaloneDir
    Copy-Item (Join-Path $root "ci\standalone-README.txt") (Join-Path $standaloneDir "README.txt")
    Copy-Item (Join-Path $root "prompts") (Join-Path $standaloneDir "prompts") -Recurse
}

# --- zip and checksum -------------------------------------------------------

Add-Type -AssemblyName System.IO.Compression.FileSystem

$archives = [ordered]@{
    "sa3-$Version-windows-x64-core.zip" = $coreDir
}
if (-not $CpuOnly) {
    $archives["sa3-$Version-windows-x64-cuda.zip"] = $cudaDir
    $archives["sa3-$Version-windows-x64-vulkan.zip"] = $vulkanDir
    $archives["sa3-$Version-windows-x64-standalone.zip"] = $standaloneDir
    if ($CudaRuntime) { $archives["cudart-$cudaMajorMinor-windows-x64.zip"] = $cudartDir }
}

$existing = @(Get-ChildItem -LiteralPath $outPath -File)
if (@($existing | Where-Object { $_.Name -notmatch '^(sa3-v\d+\.\d+\.\d+-windows-x64-(core|cuda|vulkan|standalone)\.zip|cudart-\d+\.\d+-windows-x64\.zip|SHA256SUMS)$' }).Count) {
    Fail "OutDir contains files that are not sa3.cpp package outputs: $outPath"
}
if ($existing.Count) { $existing | Remove-Item -Force }

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
