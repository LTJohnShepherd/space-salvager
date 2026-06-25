@echo off
REM Launch manual play inside the project venv. Double-click this file.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python manual_play.py
echo.
echo ---- script ended (close this window or press a key) ----
pause
