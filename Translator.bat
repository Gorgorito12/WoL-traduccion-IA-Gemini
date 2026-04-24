@echo off
REM Double-click launcher for Windows. No admin required.

cd /d "%~dp0"

REM First-run convenience: install the optional drag-and-drop package silently.
python -m pip show tkinterdnd2 >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing optional drag-and-drop support...
    python -m pip install --quiet tkinterdnd2
)

REM Prefer pythonw.exe (no console window); fall back to python.exe.
where pythonw.exe >nul 2>&1
if %errorlevel%==0 (
    start "" pythonw.exe translate_gui.py
) else (
    start "" python.exe translate_gui.py
)
