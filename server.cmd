@echo off
rem Launch sa3-server. Extra args pass through, e.g. server.cmd --model small-music --port 9000.
rem See docs\SERVER.md.
setlocal
set "BIN="
rem After env.cmd, the sa3-server on PATH is the build the user just made; prefer it.
for /f "delims=" %%p in ('where sa3-server.exe 2^>nul') do if not defined BIN set "BIN=%%p"
for %%d in (build-cuda build-vulkan build-all build) do (
    if not defined BIN if exist "%%d\bin\Release\sa3-server.exe" set "BIN=%%d\bin\Release\sa3-server.exe"
)
if "%BIN%"=="" ( echo sa3-server not built - run build.cmd ^<backend^> first& exit /b 1 )
echo [sa3-server] launching %BIN%
echo [sa3-server] LoRA training is not started here; studio.cmd starts it with the server.
"%BIN%" --model medium --encoding f16 --port 8006 %*
