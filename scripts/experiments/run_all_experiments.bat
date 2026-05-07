@echo off
echo RUNNING IC ON BA...
call uv run python scripts\experiments\train_rf_ic_ba.py

echo RUNNING IC ON ER...
call uv run python scripts\experiments\train_rf_ic_er.py

echo RUNNING SI ON BA...
call uv run python scripts\experiments\train_rf_si_ba.py

echo RUNNING SI ON ER...
call uv run python scripts\experiments\train_rf_si_er.py

echo ALL EXPERIMENTS COMPLETED!
pause
