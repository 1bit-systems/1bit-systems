@echo off
setlocal enabledelayedexpansion

REM ──────────────────────────────────────────────────────────
REM build_windows.cmd — Build 1bit-systems for Windows
REM ──────────────────────────────────────────────────────────
REM Builds portable C++23 targets with MSVC.
REM Requires: Visual Studio 2022 BuildTools
REM ──────────────────────────────────────────────────────────

set SCRIPT_DIR=%~dp0
set REPO_DIR=%SCRIPT_DIR%..
set BUILD_DIR=%REPO_DIR%\build_windows
set ALT_CMAKE=%REPO_DIR%\CMakeLists_windows.txt

echo ╔══════════════════════════════════════════╗
echo ║  1bit-systems Windows Build              ║
echo ╚══════════════════════════════════════════╝
echo.

REM ── Detect VS 2022 ──
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo [ERROR] vswhere not found
    exit /b 1
)

for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -property installationPath`) do set VSINST=%%i
if "%VSINST%"=="" (
    echo [ERROR] Visual Studio not found
    exit /b 1
)
echo [VS] %VSINST%

REM ── Set up MSVC x64 environment ──
call "%VSINST%\VC\Auxiliary\Build\vcvars64.bat" >nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to set up MSVC
    exit /b 1
)
echo [MSVC] x64 environment ready

REM ── CMake ──
where cmake >nul 2>nul
if %ERRORLEVEL% neq 0 (
    set "CMAKE_PATH=%VSINST%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
    set "PATH=%CMAKE_PATH%;%PATH%"
)
cmake --version | findstr "version"

REM ── Set Vulkan SDK path ──
if "%VULKAN_SDK%"=="" (
    if exist "C:\VulkanSDK\1.4.350.0" (
        set "VULKAN_SDK=C:\VulkanSDK\1.4.350.0"
    )
)

REM ── Prepare build ──
if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
mkdir "%BUILD_DIR%"

REM ── Copy alternate CMakeLists.txt as CMakeLists.txt ──
copy /y "%ALT_CMAKE%" "%BUILD_DIR%\CMakeLists.txt" >nul
echo [*] Using Windows CMake configuration

REM ── Configure ──
echo.
echo [*] Configuring...
cmake -B "%BUILD_DIR%" -S "%BUILD_DIR%" -G "Visual Studio 17 2022" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DUSE_VULKAN=ON ^
    -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded

if %ERRORLEVEL% neq 0 (
    echo [ERROR] CMake configuration failed
    exit /b 1
)

REM ── Build ──
echo.
echo [*] Building...
cmake --build "%BUILD_DIR%" --config Release --parallel %NUMBER_OF_PROCESSORS%

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Build failed
    exit /b 1
)

echo.
echo ╔══════════════════════════════════════════╗
echo ║  BUILD SUCCESSFUL                        ║
echo ╚══════════════════════════════════════════╝
echo.
echo Binaries:
for %%f in ("%BUILD_DIR%\Release\*.exe") do echo   %%~nxf
echo.
echo Package: %BUILD_DIR%\1bit-systems-*-windows-x64.zip

exit /b 0
