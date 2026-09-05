@echo off
setlocal enabledelayedexpansion
rem Download the sa3.cpp GGUF model set from HuggingFace (public repos) with curl.exe - no Python.
rem Usage: models.cmd [--variant medium^|small-music^|small-sfx] [--encoding TYPE] ...
rem        models.cmd --sat [--sat-model saos] [--saos-variant arc^|kickbass^|jerry-grunge] [--encoding TYPE] ...
rem   default: medium f16 DiT, f32 autoencoder, into .\models

set "VARIANT=medium"
set "ENCODING=f16"
set "ENCODING_SET=0"
rem The text encoder resolves apart from the DiT: F16 is equivalent to F32 at half the size.
set "T5_ENCODING=f16"
set "T5_ENCODING_SET=0"
rem So does the autoencoder, and it defaults to F32. SAME used to ride on --encoding, so asking for
rem a quantized DiT quietly fetched a quantized SAME too -- the last net the audio crosses, and the
rem one continuation/transform cross twice per iteration. Quantized autoencoders are still
rem published; ask for one with --ae-encoding.
set "AE_ENCODING=f32"
set "AE_ENCODING_SET=0"
set "SAT=0"
set "SAT_MODEL=saos"
set "SAOS_VARIANT=arc"
set "NAMESPACE=thepatch"
set "OUT=models"
set "TRAINING_BASE=0"
set "DRY_RUN=0"

:parse
if "%~1"=="" goto parsed
if /I "%~1"=="--variant"   ( set "VARIANT=%~2" & shift & shift & goto parse )
if /I "%~1"=="--encoding"  ( set "ENCODING=%~2" & set "ENCODING_SET=1" & shift & shift & goto parse )
if /I "%~1"=="--t5-encoding" ( set "T5_ENCODING=%~2" & set "T5_ENCODING_SET=1" & shift & shift & goto parse )
if /I "%~1"=="--ae-encoding" ( set "AE_ENCODING=%~2" & set "AE_ENCODING_SET=1" & shift & shift & goto parse )
if /I "%~1"=="--sat"       ( set "SAT=1" & shift & goto parse )
if /I "%~1"=="--sat-model" ( set "SAT_MODEL=%~2" & shift & shift & goto parse )
if /I "%~1"=="--saos-variant" ( set "SAOS_VARIANT=%~2" & shift & shift & goto parse )
if /I "%~1"=="--namespace" ( set "NAMESPACE=%~2" & shift & shift & goto parse )
if /I "%~1"=="--out"       ( set "OUT=%~2" & shift & shift & goto parse )
if /I "%~1"=="--training-base" ( set "TRAINING_BASE=1" & shift & goto parse )
if /I "%~1"=="--dry-run"   ( set "DRY_RUN=1" & shift & goto parse )
if /I "%~1"=="-h"          goto help
if /I "%~1"=="--help"      goto help
echo unknown option: %~1 & exit /b 1
:parsed

if "%SAT%"=="1" goto sat_setup

if /I "%VARIANT%"=="medium"      ( set "DIT_SIZE=1.5B" & set "SAME=same-l" & goto variant_ok )
if /I "%VARIANT%"=="small-music" ( set "DIT_SIZE=0.5B" & set "SAME=same-s" & goto variant_ok )
if /I "%VARIANT%"=="small-sfx"   ( set "DIT_SIZE=0.5B" & set "SAME=same-s" & goto variant_ok )
echo unknown variant: %VARIANT% ^(medium^|small-music^|small-sfx^) & exit /b 1
:variant_ok
rem Training bases are published F16, F32 and Q4_K_M -- training on a quantized base works on every
rem backend. Q5_K_M/Q8_0 bases are not published, so those pull an F16 base DiT. The [fetch] lines
rem show which one was resolved.
set "BASE_ENC="
if /I "%ENCODING%"=="f16"    ( set "ENC=F16"     & goto encoding_ok )
if /I "%ENCODING%"=="f32"    ( set "ENC=F32"     & goto encoding_ok )
if /I "%ENCODING%"=="q4_k_m" ( set "ENC=Q4_K_M"  & goto encoding_ok )
if /I "%ENCODING%"=="q5_k_m" ( set "ENC=Q5_K_M"  & set "BASE_ENC=F16" & goto encoding_ok )
if /I "%ENCODING%"=="q8_0"   ( set "ENC=Q8_0"    & set "BASE_ENC=F16" & goto encoding_ok )
echo unknown encoding: %ENCODING% ^(f16^|f32^|q4_k_m^|q5_k_m^|q8_0^) & exit /b 1
:encoding_ok
if not defined BASE_ENC set "BASE_ENC=%ENC%"
set "AE_ENC="
if /I "%AE_ENCODING%"=="f16"    set "AE_ENC=F16"
if /I "%AE_ENCODING%"=="f32"    set "AE_ENC=F32"
if /I "%AE_ENCODING%"=="q4_k_m" set "AE_ENC=Q4_K_M"
if /I "%AE_ENCODING%"=="q5_k_m" set "AE_ENC=Q5_K_M"
if /I "%AE_ENCODING%"=="q8_0"   set "AE_ENC=Q8_0"
if not defined AE_ENC (
    echo unknown --ae-encoding "%AE_ENCODING%" ^(expected f16^|f32^|q4_k_m^|q5_k_m^|q8_0^) 1>&2
    exit /b 2
)
set "T5_ENC="
if /I "%T5_ENCODING%"=="f16"  set "T5_ENC=F16"
if /I "%T5_ENCODING%"=="f32"  set "T5_ENC=F32"
if /I "%T5_ENCODING%"=="q8_0" set "T5_ENC=Q8_0"
if not defined T5_ENC (
    echo unknown --t5-encoding "%T5_ENCODING%" ^(expected f16^|f32^|q8_0^) 1>&2
    exit /b 2
)
set "VAR_REPO=%NAMESPACE%/stable-audio-3-%VARIANT%-GGUF"
set "BASE_REPO=%NAMESPACE%/stable-audio-3-%VARIANT%-base-GGUF"
set "SHARED=%NAMESPACE%/t5gemma-b-b-ul2-GGUF"
set "BASE=stable-audio-3-%VARIANT%"
if not exist "%OUT%" mkdir "%OUT%"

call :dl "%VAR_REPO%" "%BASE%-dit-%DIT_SIZE%-v1.0-%ENC%.gguf"
if errorlevel 1 exit /b 1
if "%TRAINING_BASE%"=="1" (
    call :dl "%BASE_REPO%" "%BASE%-base-dit-%DIT_SIZE%-v1.0-%BASE_ENC%.gguf"
    if errorlevel 1 exit /b 1
)
call :dl "%VAR_REPO%" "%BASE%-%SAME%-v1.0-%AE_ENC%.gguf"
if errorlevel 1 exit /b 1
call :dl "%VAR_REPO%" "%BASE%-conditioner-v1.0-F32.gguf"
if errorlevel 1 exit /b 1
call :dl "%SHARED%"   "t5gemma-b-b-ul2-encoder-0.3B-v1.0-%T5_ENC%.gguf"
if errorlevel 1 exit /b 1
call :dl "%SHARED%"   "t5gemma-b-b-ul2-v1.0-vocab.gguf"
if errorlevel 1 exit /b 1
if "%TRAINING_BASE%"=="1" (
    echo [done] %VARIANT% ^(DiT %ENC%, SAME %AE_ENC%^) + training base -^> %OUT%\
) else (
    echo [done] %VARIANT% ^(DiT %ENC%, SAME %AE_ENC%^) -^> %OUT%\
)
exit /b 0

:sat_setup
if not "%TRAINING_BASE%"=="0" (
    echo --training-base applies to SA3, not --sat 1>&2
    exit /b 2
)
if /I not "%SAT_MODEL%"=="saos" (
    echo unknown --sat-model "%SAT_MODEL%" ^(currently: saos^) 1>&2
    exit /b 2
)
if "%ENCODING_SET%"=="0" set "ENCODING=q5_k_m"
if "%T5_ENCODING_SET%"=="0" set "T5_ENCODING=%ENCODING%"
if "%AE_ENCODING_SET%"=="0" set "AE_ENCODING=%ENCODING%"
set "ENC="
if /I "%ENCODING%"=="f16" set "ENC=F16"
if /I "%ENCODING%"=="q8_0" set "ENC=Q8_0"
if /I "%ENCODING%"=="q5_k_m" set "ENC=Q5_K_M"
if /I "%ENCODING%"=="q4_k_m" set "ENC=Q4_K_M"
if not defined ENC ( echo unsupported SAOS encoding "%ENCODING%" 1>&2 & exit /b 2 )
set "T5_ENC="
if /I "%T5_ENCODING%"=="f16" set "T5_ENC=F16"
if /I "%T5_ENCODING%"=="q8_0" set "T5_ENC=Q8_0"
if /I "%T5_ENCODING%"=="q5_k_m" set "T5_ENC=Q5_K_M"
if /I "%T5_ENCODING%"=="q4_k_m" set "T5_ENC=Q4_K_M"
if not defined T5_ENC ( echo unsupported SAOS T5 encoding "%T5_ENCODING%" 1>&2 & exit /b 2 )
set "AE_ENC="
if /I "%AE_ENCODING%"=="f16" set "AE_ENC=F16"
if /I "%AE_ENCODING%"=="q8_0" set "AE_ENC=Q8_0"
if /I "%AE_ENCODING%"=="q5_k_m" set "AE_ENC=Q5_K_M"
if /I "%AE_ENCODING%"=="q4_k_m" set "AE_ENC=Q4_K_M"
if not defined AE_ENC ( echo unsupported SAOS Oobleck encoding "%AE_ENCODING%" 1>&2 & exit /b 2 )
if /I "%SAOS_VARIANT%"=="arc" set "SAOS_DIT=stable-audio-open-small-dit-0.3B-v1.0-%ENC%.gguf"
if /I "%SAOS_VARIANT%"=="kickbass" set "SAOS_DIT=finetunes/kickbass/kickbass-v1-e257-dit-0.3B-v1.0-%ENC%.gguf"
if /I "%SAOS_VARIANT%"=="jerry-grunge" set "SAOS_DIT=finetunes/jerry-grunge/jerry-grunge-bs64-step3000-dit-0.3B-v1.0-%ENC%.gguf"
if not defined SAOS_DIT ( echo unknown SAOS variant "%SAOS_VARIANT%" 1>&2 & exit /b 2 )
set "SAOS_REPO=%NAMESPACE%/stable-audio-open-small-GGUF"
if not exist "%OUT%" mkdir "%OUT%"
call :dl "%SAOS_REPO%" "%SAOS_DIT%"
if errorlevel 1 exit /b 1
call :dl "%SAOS_REPO%" "t5-base-encoder-0.1B-v1.0-%T5_ENC%.gguf"
if errorlevel 1 exit /b 1
call :dl "%SAOS_REPO%" "stable-audio-open-small-oobleck-v1.0-%AE_ENC%.gguf"
if errorlevel 1 exit /b 1
echo [done] SAOS %SAOS_VARIANT% ^(%ENC%^) -^> %OUT%\
exit /b 0

:help
echo Usage: models.cmd [SA3 options] or --sat [--sat-model saos] [--saos-variant arc^|kickbass^|jerry-grunge] [--encoding TYPE] [--out DIR] [--dry-run]
exit /b 0

:dl
set "DST=%OUT%\%~2"
for %%D in ("%DST%") do if not exist "%%~dpD" mkdir "%%~dpD"
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
