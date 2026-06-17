@echo off
REM ===== SherdHunter dashboard launcher =====
REM Double-click to open the viewer. Starts the local server only if it isn't already running.
title SherdHunter Dashboard
cd /d "%~dp0"

REM is something already serving on 8753?
python -c "import socket,sys; s=socket.socket(); r=s.connect_ex(('127.0.0.1',8753)); s.close(); sys.exit(0 if r==0 else 1)" 2>nul
if errorlevel 1 (
  echo Starting SherdHunter server on http://localhost:8753 ...
  start "SherdHunter server" /min python -m http.server 8753 --directory viewer
  python -c "import time; time.sleep(1.3)"
) else (
  echo SherdHunter server already running.
)

start "" "http://localhost:8753"
exit /b 0
