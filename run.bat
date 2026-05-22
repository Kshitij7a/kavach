@echo off
echo Starting Project Kavach...
call .venv\Scripts\activate.bat
set PYTHONPATH=src
python -m kavach.orchestrator.main
pause