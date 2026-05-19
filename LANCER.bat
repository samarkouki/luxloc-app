@echo off
echo ============================================
echo    LUXLOC - Lancement de l'application
echo ============================================
echo.
echo Installation des dependances...
pip install fastapi uvicorn sqlalchemy pydantic "passlib[bcrypt]" python-multipart
echo.
echo Lancement du serveur...
echo.
echo ============================================
echo    Ouvre ton navigateur sur :
echo    http://localhost:8000
echo ============================================
echo.
uvicorn main:app --reload --port 8000
pause
