@echo off
setlocal enabledelayedexpansion
rem Download the sa3.cpp GGUF model set from HuggingFace (public repos) with curl.exe - no Python.
rem Usage: models.cmd [--variant medium^|small-music^|small-sfx] [--encoding f16^|f32] [--training-base] [--namespace <hf-user>] [--out DIR] [--dry-run]
rem   default: medium f16 into .\models

set "VARIANT=medium"
set "ENCODING=f16"
set "NAMESPACE=thepatch"
set "OUT=models"
set "TRAINING_BASE=0"
set "DRY_RUN=0"

:parse
if "%~1"=="" goto parsed
if /I "%~1"=="--variant"   ( set "VARIANT=%~2" & shift & shift & goto parse )
if /I "%~1"=="--encoding"  ( set "ENCODING=%~2" & shift & shift & goto parse )
if /I "%~1"=="--namespace" ( set "NAMESPACE=%~2" & shift & shift & goto parse )
if /I "%~1"=="--out"       ( set "OUT=%~2" & shift & shift & goto parse )
if /I "%~1"=="--training-base" ( set "TRAINING_BASE=1" & shift & goto parse )
if /I "%~1"=="--dry-run"   ( set "DRY_RUN=1" & shift & goto parse )
if /I "%~1"=="-h"          goto help
if /I "%~1"=="--help"      goto help
echo unknown option: %~1 & exit /b 1
:parsed

if /I "%VARIANT%"=="medium"      ( set "DIT_SIZE=1.5B" & set "SAME=same-l" & goto variant_ok )
if /I "%VARIANT%"=="small-music" ( set "DIT_SIZE=0.5B" & set "SAME=same-s" & goto variant_ok )
if /I "%VARIANT%"=="small-sfx"   ( set "DIT_SIZE=0.5B" & set "SAME=same-s" & goto variant_ok )
echo unknown variant: %VARIANT% ^(medium^|small-music^|small-sfx^) & exit /b 1
:variant_ok
if /I "%ENCODING%"=="f16" ( set "ENC=F16" & goto encoding_ok )
if /I "%ENCODING%"=="f32" ( set "ENC=F32" & goto encoding_ok )
echo unknown encoding: %ENCODING% ^(f16^|f32^) & exit /b 1
:encoding_ok
set "VAR_REPO=%NAMESPACE%/stable-audio-3-%VARIANT%-GGUF"
set "BASE_REPO=%NAMESPACE%/stable-audio-3-%VARIANT%-base-GGUF"
set "SHARED=%NAMESPACE%/t5gemma-b-b-ul2-GGUF"
set "BASE=stable-audio-3-%VARIANT%"
if not exist "%OUT%" mkdir "%OUT%"

call :dl "%VAR_REPO%" "%BASE%-dit-%DIT_SIZE%-v1.0-%ENC%.gguf"
if errorlevel 1 exit /b 1
if "%TRAINING_BASE%"=="1" (
    call :dl "%BASE_REPO%" "%BASE%-base-dit-%DIT_SIZE%-v1.0-%ENC%.gguf"
    if errorlevel 1 exit /b 1
)
call :dl "%VAR_REPO%" "%BASE%-%SAME%-v1.0-%ENC%.gguf"
if errorlevel 1 exit /b 1
call :dl "%VAR_REPO%" "%BASE%-conditioner-v1.0-F32.gguf"
if errorlevel 1 exit /b 1
call :dl "%SHARED%"   "t5gemma-b-b-ul2-encoder-0.3B-v1.0-F32.gguf"
if errorlevel 1 exit /b 1
call :dl "%SHARED%"   "t5gemma-b-b-ul2-v1.0-vocab.gguf"
if errorlevel 1 exit /b 1
if "%TRAINING_BASE%"=="1" (
    echo [done] %VARIANT% ^(%ENCODING%^) + training base -^> %OUT%\
) else (
    echo [done] %VARIANT% ^(%ENCODING%^) -^> %OUT%\
)
exit /b 0

:help
echo Usage: models.cmd [--variant medium^|small-music^|small-sfx] [--encoding f16^|f32] [--training-base] [--namespace HF_USER] [--out DIR] [--dry-run]
exit /b 0

:dl
set "DST=%OUT%\%~2"
if "%DRY_RUN%"=="1" (
    echo [plan] https://huggingface.co/%~1/resolve/main/%~2 -^> %DST%
    exit /b 0
)
if exist "%DST%" (
    echo [check/resume] %~2
) else (
    echo [download] %~1/%~2
)
curl.exe -fL --retry 3 --continue-at - -o "%DST%" "https://huggingface.co/%~1/resolve/main/%~2"
if errorlevel 1 (
    echo [error] failed to download %~1/%~2
    exit /b 1
)
exit /b 0
