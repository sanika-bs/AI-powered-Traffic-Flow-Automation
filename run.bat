@echo off
REM One-click launcher for Windows: creates venv on first run, installs deps, starts the app.
cd /d "%~dp0"
if not exist venv (
  echo Creating virtual environment...
  py -3 -m venv venv || python -m venv venv
  call venv\Scripts\activate.bat
  python -m pip install --upgrade pip
  pip install -r requirements.txt
) else (
  call venv\Scripts\activate.bat
)
start "" http://127.0.0.1:5000
python app.py
pause
