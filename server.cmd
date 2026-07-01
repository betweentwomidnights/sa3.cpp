@echo off
rem Launch sa3-server. Extra args pass through, e.g. server.cmd --model small-music --port 9000.
rem See docs\SERVER.md.
setlocal
set "BIN="
for %%d in (build-cuda build-vulkan build-all build) do (
    if not defined BIN if exist "%%d\bin\Release\sa3-server.exe" set "BIN=%%d\bin\Release\sa3-server.exe"
)
if "%BIN%"=="" ( echo sa3-server not built - run build.cmd ^<backend^> first& exit /b 1 )
echo [sa3-server] launching %BIN%
"%BIN%" --model medium --encoding f16 --port 8086 %*
