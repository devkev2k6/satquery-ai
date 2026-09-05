@echo off
rem ==============================================================================
rem SatQuery AI - Environment Setup Script (Windows)
rem Creates a Python virtual environment (.venv) and installs project dependencies
rem ==============================================================================

setlocal enabledelayedexpansion

echo ==========================================
echo SatQuery AI - Virtual Environment Setup (Windows)
echo ==========================================

cd /d "%~dp0\.."

rem Locate Python executable
set "PYTHON_EXE="

py -3 --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PYTHON_EXE=py -3"
    goto :found_python
)

python --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PYTHON_EXE=python"
    goto :found_python
)

python3 --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PYTHON_EXE=python3"
    goto :found_python
)

if exist "%APPDATA%\uv\python\cpython-3.11-windows-x86_64-none\python.exe" (
    set "PYTHON_EXE=%APPDATA%\uv\python\cpython-3.11-windows-x86_64-none\python.exe"
    goto :found_python
)

if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    goto :found_python
)

echo [-] Error: Python was not found in your PATH.
echo     Please install Python 3.9+ from https://www.python.org/downloads/
echo     Make sure to check "Add python.exe to PATH" during installation.
exit /b 1

:found_python
echo [+] Using Python command: !PYTHON_EXE!
!PYTHON_EXE! --version

set "VENV_DIR=.venv"

if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [*] Virtual environment already exists at %VENV_DIR%
) else (
    echo [*] Creating virtual environment at %VENV_DIR% ...
    !PYTHON_EXE! -m venv "%VENV_DIR%"
    if %ERRORLEVEL% neq 0 (
        echo [-] Failed to create virtual environment.
        exit /b %ERRORLEVEL%
    )
    echo [+] Virtual environment created successfully.
)

echo [*] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

echo [*] Upgrading pip...
python -m pip install --upgrade pip

if exist "requirements.txt" (
    echo [*] Installing dependencies from requirements.txt...
    pip install -r requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo [-] Encountered an error installing packages.
        exit /b %ERRORLEVEL%
    )
    echo [+] Dependencies installed successfully.
) else (
    echo [-] Warning: requirements.txt not found.
)

echo ==========================================
echo Environment setup complete!
echo To activate this environment in Windows cmd / powershell, run:
echo     .venv\Scripts\activate
echo ==========================================
endlocal
