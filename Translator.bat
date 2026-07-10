@echo off
setlocal EnableExtensions
REM Double-click launcher for Windows. No admin required.
REM Self-sufficient: installs Python (via winget) and the pip dependencies if missing.

cd /d "%~dp0"

call :find_python
if defined PY goto have_python

echo.
echo Python no esta instalado. Se instalara automaticamente con winget (~2 min)...
echo (Python is not installed. Installing automatically via winget...)
echo.
where winget >nul 2>&1
if errorlevel 1 goto no_winget
winget install -e --id Python.Python.3.13 --accept-source-agreements --accept-package-agreements
call :find_python
if defined PY goto have_python
echo.
echo ERROR: La instalacion automatica de Python fallo. (Automatic install failed.)
goto manual_msg

:have_python
echo Usando Python: %PY%

REM tkinter ships with the python.org installer; verify anyway.
"%PY%" -c "import tkinter" >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Esta instalacion de Python no incluye tkinter, necesario para la interfaz.
    echo Reinstala Python desde https://www.python.org/downloads/ con las opciones por defecto.
    pause
    exit /b 1
)

REM Core dependencies for translate_gemini.py.
"%PY%" -c "import google.genai, tqdm" >nul 2>&1
if errorlevel 1 (
    echo Instalando dependencias: google-genai, tqdm...
    "%PY%" -m pip install --quiet google-genai tqdm
)
"%PY%" -c "import google.genai, tqdm" >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: No se pudieron instalar las dependencias de pip: google-genai, tqdm.
    echo Revisa tu conexion a internet y vuelve a ejecutar este archivo.
    pause
    exit /b 1
)

REM Optional drag-and-drop support.
"%PY%" -c "import tkinterdnd2" >nul 2>&1
if errorlevel 1 (
    echo Instalando soporte opcional de arrastrar y soltar: tkinterdnd2...
    "%PY%" -m pip install --quiet tkinterdnd2
)

REM Prefer the sibling pythonw.exe (no console window); fall back to python.exe.
if defined PYW (
    start "" "%PYW%" translate_gui.py
) else (
    start "" "%PY%" translate_gui.py
)
exit /b 0

:no_winget
echo winget no esta disponible en este sistema. (winget is not available.)
:manual_msg
echo.
echo Instala Python manualmente desde https://www.python.org/downloads/
echo marcando la casilla "Add python.exe to PATH", y vuelve a ejecutar este archivo.
echo (Install Python from python.org, check "Add python.exe to PATH", then re-run this file.)
pause
exit /b 1

REM ---------------------------------------------------------------------------
REM find_python: sets PY to a working python.exe and PYW to its sibling
REM pythonw.exe (if any). The Microsoft Store alias stub fails the `-c` probe,
REM so it is never selected.
:find_python
set "PY="
set "PYW="
where py >nul 2>&1
if not errorlevel 1 (
    py -3 -c "import sys" >nul 2>&1
    if not errorlevel 1 (
        for /f "delims=" %%I in ('py -3 -c "import sys; print(sys.executable)"') do set "PY=%%I"
    )
)
if not defined PY (
    python -c "import sys" >nul 2>&1
    if not errorlevel 1 (
        for /f "delims=" %%I in ('python -c "import sys; print(sys.executable)"') do set "PY=%%I"
    )
)
if not defined PY (
    for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
        if exist "%%D\python.exe" set "PY=%%D\python.exe"
    )
)
if defined PY for %%I in ("%PY%") do if exist "%%~dpIpythonw.exe" set "PYW=%%~dpIpythonw.exe"
exit /b 0
